"""
FastAPI Application for Credit Scoring Model with MLflow Tracking

This API exposes a pre-trained credit scoring model for predictions.
The model is loaded once at startup and reused for all requests.

MLflow Integration:
- All predictions are logged to MLflow
- Request metadata, input features, and predictions are tracked
- Model metadata (run ID, version) is included in responses
"""

import logging
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING, cast

import joblib
import numpy as np
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

from api.ml_pipeline import setup_mlflow_experiment, compute_form_coverage


class JsonLogFormatter(logging.Formatter):
    """Format log records as one JSON object per line (structured logging).

    Machine-parsable production logs: each line carries the timestamp, the
    level, the logger and the message, so the logs can be collected and
    analyzed without regex parsing (see monitoring/DRIFT_STUDY.md).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# Configure logging. Structured JSON by default (as recommended for
# production log analysis); LOG_FORMAT=text restores the classic
# human-readable format for local debugging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
if os.getenv("LOG_FORMAT", "json").lower() != "text":
    for _handler in logging.getLogger().handlers:
        _handler.setFormatter(JsonLogFormatter())


# Load environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "api/models/credit_scoring_model.pkl")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", str(Path(__file__).parent / "schema" / "raw_columns.json"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///tracking/mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "credit_scoring_api")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH",
                            str(Path(__file__).parent / "models" / "credit_scoring_model.onnx"))
# Inference engine: "auto" (ONNX when available, LightGBM otherwise),
# "onnx" or "lightgbm". See monitoring/PERF_STUDY.md for the benchmark.
INFERENCE_ENGINE = os.getenv("INFERENCE_ENGINE", "auto").lower()
# MLflow prediction logging: "async" (default, off the request critical
# path: the sqlite logging measured ~255 ms/prediction), "sync" (blocking,
# the pre-optimization behavior) or "off" (no per-prediction MLflow runs;
# the request JSONL log and /monitoring keep working).
MLFLOW_LOG_MODE = os.getenv("MLFLOW_LOG_MODE", "async").lower()

# Configure MLflow at import time (not only at startup): predictions can be
# logged before the startup event runs (e.g. TestClient without a context
# manager in tests), and the default ./mlruns location would then scatter
# artifacts at the project root.
setup_mlflow_experiment(MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI)

# Raw input schema (generated at training time by api/train_model.py)
with open(SCHEMA_PATH, encoding="utf-8") as _f:
    SCHEMA = json.load(_f)
RAW_FIELDS: List[Dict[str, Any]] = SCHEMA["fields"]

# Global variables
model = None
preprocessor = None
model_version = "1.0.0"
feature_names = None
mlflow_run_id = None
model_metadata = None
onnx_session = None


# ============================================================================
# Inference engine (ONNX Runtime preferred, LightGBM fallback)
# ============================================================================

def load_onnx_session():
    """Create the ONNX Runtime session if requested and available.

    Graceful fallback to LightGBM (with a warning) in "auto" mode when the
    .onnx artifact or the onnxruntime package is missing; a hard warning in
    explicit "onnx" mode. Serving must not break because of the engine.
    """
    global onnx_session
    if INFERENCE_ENGINE == "lightgbm":
        return
    if not Path(ONNX_MODEL_PATH).exists():
        if INFERENCE_ENGINE == "onnx":
            logger.warning(f"ONNX model not found at {ONNX_MODEL_PATH}: "
                           "falling back to the LightGBM model")
        return
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(ONNX_MODEL_PATH,
                                        providers=["CPUExecutionProvider"])
    except ImportError:
        if INFERENCE_ENGINE == "onnx":
            logger.warning("onnxruntime is not installed: falling back to "
                           "the LightGBM model (uv sync --extra onnx)")
        return
    except Exception as e:
        logger.warning(f"ONNX session could not be created ({e}): "
                       "falling back to the LightGBM model")
        return
    # The session input width must match the preprocessor output: a pickle
    # and an .onnx regenerated at different times would otherwise silently
    # produce wrong predictions.
    expected = session.get_inputs()[0].shape[-1]
    if feature_names is not None and expected not in (None, len(feature_names)):
        logger.warning(f"ONNX input width ({expected}) != preprocessor output "
                       f"({len(feature_names)}): falling back to the LightGBM model")
        return
    onnx_session = session
    logger.info(f"ONNX Runtime session loaded: {ONNX_MODEL_PATH} "
                f"({expected} features)")


def predict_probability(input_df) -> float:
    """P(default) for a single preprocessed row, via the active engine.

    ONNX Runtime path: ~0.4 ms/prediction vs ~4 ms for LightGBM
    (monitoring/PERF_STUDY.md). The prediction class is derived from the
    probability (> 0.5, LightGBM's own decision rule) instead of calling
    model.predict a second time.
    """
    if onnx_session is not None:
        out_names = [o.name for o in onnx_session.get_outputs()]
        output_name = "probabilities" if "probabilities" in out_names else out_names[-1]
        X = input_df.to_numpy().astype(np.float32)
        proba = onnx_session.run([output_name],
                                 {onnx_session.get_inputs()[0].name: X})[0]
        return float(proba[0][1])
    if model is None:
        raise RuntimeError("No inference engine available (model not loaded)")
    return float(model.predict_proba(input_df)[0][1])


def warm_up_inference() -> None:
    """Run one full prediction at startup to pay the cold-start cost
    (imports, LightGBM first pass, ONNX session init) before real traffic
    arrives. The monitoring data showed a ~1.2 s first request in
    production; it now happens at deployment time.
    """
    if model is None or preprocessor is None:
        return
    try:
        dummy = pd.DataFrame([{f["name"]: f["default"] for f in RAW_FIELDS}])
        predict_probability(preprocessor.transform(dummy))
        logger.info("Inference warm-up completed")
    except Exception as e:
        logger.warning(f"Warm-up skipped ({e})")


# ============================================================================
# Background MLflow logging
# ============================================================================

# Single worker: sqlite cannot take concurrent writers, and serializing the
# runs (~255 ms each, measured) is far above the production traffic rate.
# Non-daemon worker threads are joined at interpreter exit, so batch tools
# (monitoring/generate_traffic.py) still see every run before exiting.
_mlflow_executor: Optional[ThreadPoolExecutor] = None


def dispatch_prediction_logging(input_data: Dict[str, Any], prediction: int,
                                probability: float, model_version: str,
                                risk_category: str, processing_time: float) -> None:
    """Send the MLflow logging off the request critical path.

    MLFLOW_LOG_MODE: async (default) | sync | off — see api/main.py header.
    """
    if MLFLOW_LOG_MODE == "off":
        return
    if MLFLOW_LOG_MODE == "sync":
        log_prediction_to_mlflow(input_data, prediction, probability,
                                 model_version, risk_category, processing_time)
        return
    global _mlflow_executor
    if _mlflow_executor is None:
        _mlflow_executor = ThreadPoolExecutor(max_workers=1,
                                              thread_name_prefix="mlflow-log")
    _mlflow_executor.submit(log_prediction_to_mlflow, input_data, prediction,
                            probability, model_version, risk_category,
                            processing_time)


# ============================================================================
# In-Memory Monitoring Buffer (served on GET /monitoring)
# ============================================================================

# Ring buffer of recent predictions. In-memory only: reset on restart or
# redeploy (no persistent disk on the Render free plan). This powers the
# dashboard's monitoring tab with real serving data.
MONITORING_BUFFER_SIZE = 500
_recent_predictions: deque = deque(maxlen=MONITORING_BUFFER_SIZE)
_prediction_lock = threading.Lock()


def record_prediction(probability: float, prediction: int,
                      risk_category: str, processing_time_ms: float) -> None:
    """Append a prediction to the in-memory monitoring buffer."""
    with _prediction_lock:
        _recent_predictions.append({
            "probability": round(float(probability), 4),
            "prediction": int(prediction),
            "risk_category": risk_category,
            "processing_time_ms": round(float(processing_time_ms), 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# Request/error counters for the operational metrics served on /monitoring.
# In-memory like the prediction buffer (the Render free plan has no durable
# disk): enough to expose a live error rate next to the latency metrics.
_request_stats = {"total": 0, "errors_4xx": 0, "errors_5xx": 0}


def record_request(status_code: Optional[int]) -> None:
    """Count a served request; status_code None means an unhandled exception (5xx)."""
    with _prediction_lock:
        _request_stats["total"] += 1
        if status_code is None or status_code >= 500:
            _request_stats["errors_5xx"] += 1
        elif status_code >= 400:
            _request_stats["errors_4xx"] += 1


# Persistent request log (JSON Lines): one structured line per served
# request — the durable counterpart of the in-memory counters above. This
# is what makes the error rate and the request latency analyzable a
# posteriori (monitoring/drift_analysis.py --requests-file). Lives under
# tracking/ by default (ephemeral on the Render free plan, like the MLflow
# store: export before redeploying, or set REQUESTS_LOG_FILE elsewhere).
REQUESTS_LOG_FILE = os.getenv("REQUESTS_LOG_FILE", os.path.join("tracking", "api_requests.jsonl"))
_requests_log_lock = threading.Lock()
_requests_log_disabled = False


def record_request_event(method: str, path: str, status_code: Optional[int],
                         duration_ms: float) -> None:
    """Append one JSON line describing a served request to REQUESTS_LOG_FILE.

    Silently disabled after the first write failure (e.g. read-only
    filesystem): monitoring must never break request serving.
    """
    global _requests_log_disabled
    if _requests_log_disabled or not REQUESTS_LOG_FILE:
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(float(duration_ms), 2),
    }
    try:
        with _requests_log_lock:
            os.makedirs(os.path.dirname(REQUESTS_LOG_FILE) or ".", exist_ok=True)
            with open(REQUESTS_LOG_FILE, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(event) + "\n")
    except OSError as e:
        _requests_log_disabled = True
        logger.warning(f"Request log disabled ({REQUESTS_LOG_FILE} not writable): {e}")


def _percentile(sorted_values: List[float], p: float) -> float:
    """Linear-interpolated percentile on an already-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def get_feature_importances(top_n: int = 20) -> List[Dict[str, Any]]:
    """Extract the model's real feature importances (top N).

    Works for LightGBM (gain-based importances via booster_) and estimators
    exposing feature_importances_ (e.g. RandomForest). Returns [] when the
    loaded model exposes no importance information.
    """
    if model is None or not feature_names:
        return []
    try:
        importances = None
        if hasattr(model, "booster_") and hasattr(model.booster_, "feature_importance"):
            importances = model.booster_.feature_importance(importance_type="gain")
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        if importances is None:
            return []
        importances = list(map(float, importances))
        if len(importances) != len(feature_names):
            logger.warning("Feature importances length mismatch: skipping")
            return []
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
        total = sum(v for _, v in pairs) or 1.0
        return [
            {"feature": name, "importance": round(value, 4),
             "importance_pct": round(100.0 * value / total, 2)}
            for name, value in pairs
        ]
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return []


# ============================================================================
# MLflow Configuration
# ============================================================================


# Input fields logged as MLflow params (the full input is logged as a JSON artifact)
KEY_PARAMS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CODE_GENDER", "CNT_CHILDREN",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "OWN_CAR_AGE", "REGION_RATING_CLIENT",
]


def log_prediction_to_mlflow(input_data: Dict[str, Any], 
                            prediction: int, 
                            probability: float, 
                            model_version: str,
                            risk_category: str,
                            processing_time: float):
    """
    Log a single prediction to MLflow.
    
    Args:
        input_data: Dictionary of raw input fields
        prediction: Predicted class (0 or 1)
        probability: Predicted probability
        model_version: Version of the model used
        risk_category: Risk category (LOW_RISK, MEDIUM_RISK, HIGH_RISK)
        processing_time: Time taken to process the request in seconds
    """
    try:
        with mlflow.start_run(run_name=f"api_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"):
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow.set_tag("source", "api")
            mlflow.set_tag("predictor", "credit_scoring")
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("risk_category", risk_category)
            
            # Log key input features as params (full input as JSON artifact)
            for key in KEY_PARAMS:
                if key in input_data and input_data[key] is not None:
                    mlflow.log_param(key, input_data[key])
            mlflow.log_dict(input_data, "input.json")
            
            # Log metrics
            mlflow.log_metric("prediction", prediction)
            mlflow.log_metric("probability", probability)
            mlflow.log_metric("processing_time_ms", processing_time * 1000)
            
            # Log model metadata
            if model_metadata:
                for key, value in model_metadata.items():
                    if isinstance(value, (int, float, str)):
                        mlflow.set_tag(f"model_{key}", str(value))
                        
    except Exception as e:
        logger.warning(f"Could not log prediction to MLflow: {str(e)}")


# ============================================================================
# Pydantic Models
# ============================================================================

# Business ranges for the raw numeric fields. Constraints are enforced by
# Pydantic only when the field is provided in a request (all fields stay
# optional: omitted fields are imputed with the training median/mode).
# Bounds follow the Home Credit application_train distributions plus
# common sense (a borrower must be of legal age, external scores are
# normalized in [0, 1], amounts cannot be negative...).
NUMERIC_CONSTRAINTS: Dict[str, Dict[str, float]] = {
    "DAYS_BIRTH": {"ge": -40015, "le": -6570},      # age 18-110 years (days, negative)
    "DAYS_EMPLOYED": {"ge": -35000, "le": 365243},  # 365243 = pensioner placeholder
    "CNT_CHILDREN": {"ge": 0, "le": 20},
    "AMT_INCOME_TOTAL": {"ge": 0},
    "AMT_CREDIT": {"ge": 0},
    "AMT_ANNUITY": {"ge": 0},
    "AMT_GOODS_PRICE": {"ge": 0},
    "EXT_SOURCE_1": {"ge": 0, "le": 1},
    "EXT_SOURCE_2": {"ge": 0, "le": 1},
    "EXT_SOURCE_3": {"ge": 0, "le": 1},
    "OWN_CAR_AGE": {"ge": 0, "le": 100},
    "DAYS_ID_PUBLISH": {"ge": -30000, "le": 0},
    "DAYS_LAST_PHONE_CHANGE": {"ge": -5000, "le": 0},
    "AMT_REQ_CREDIT_BUREAU_YEAR": {"ge": 0, "le": 50},
    "REGION_RATING_CLIENT": {"ge": 1, "le": 3},
}


def _build_prediction_input(fields: List[Dict[str, Any]]):
    """Build the Pydantic input model dynamically from the raw schema.

    All fields are optional: any field omitted in a request is filled with
    the training median (numeric) or mode (categorical) before prediction.
    Numeric fields listed in NUMERIC_CONSTRAINTS carry business range
    constraints (ge/le): out-of-range values are rejected with a 422 before
    reaching the model.
    """
    attrs: Dict[str, Any] = {}
    for f in fields:
        if f["type"] == "categorical":
            attrs[f["name"]] = (Optional[str], Field(default=None, description=f["description"]))
        else:
            attrs[f["name"]] = (
                Optional[float],
                Field(
                    default=None,
                    description=f["description"],
                    # typed as Any: mypy cannot match Field's overloads when
                    # unpacking the constraint dict
                    **cast(Dict[str, Any], NUMERIC_CONSTRAINTS.get(f["name"], {})),
                ),
            )
    return create_model("PredictionInput", **attrs)


# Input schema aligned on the raw Home Credit columns used at training time.
# The class is created dynamically from the raw schema; the decoy under
# TYPE_CHECKING only exists for static type checkers (mypy/Pylance),
# which cannot use a variable as a type annotation.
if TYPE_CHECKING:

    class PredictionInput(BaseModel):
        """Raw application input, dynamically defined at runtime."""
else:
    PredictionInput = _build_prediction_input(RAW_FIELDS)


class PredictionOutput(BaseModel):
    """Output schema for credit scoring prediction."""
    prediction: int = Field(..., description="Predicted class (0=No Default, 1=Default)")
    probability: float = Field(..., description="Predicted probability of default (0.0-1.0)")
    risk_category: str = Field(..., description="Risk category based on probability")
    model_version: str = Field(..., description="Version of the model used")
    model_type: Optional[str] = Field(None, description="Type of the model")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID used for training")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthCheckOutput(BaseModel):
    """Output schema for health check endpoint."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    preprocessor_loaded: bool = Field(..., description="Whether the input preprocessor is loaded")
    model_version: Optional[str] = Field(None, description="Version of the loaded model")
    model_type: Optional[str] = Field(None, description="Type of the loaded model")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    feature_count: Optional[int] = Field(None, description="Number of features")


class MonitoringOutput(BaseModel):
    """Live serving metrics from the in-memory prediction buffer."""
    total_predictions: int = Field(..., description="Predictions currently in the buffer")
    buffer_capacity: int = Field(..., description="Maximum records kept in memory")
    total_requests: int = Field(0, description="Requests served since the API started")
    errors_4xx: int = Field(0, description="Client errors (4xx: validation, not found...) since start")
    errors_5xx: int = Field(0, description="Server errors (5xx: unhandled exception, degraded model...) since start")
    error_rate: Optional[float] = Field(None, description="Fraction of requests that failed (4xx + 5xx)")
    avg_latency_ms: Optional[float] = Field(None, description="Average inference time (ms)")
    p95_latency_ms: Optional[float] = Field(None, description="95th percentile inference time (ms)")
    min_latency_ms: Optional[float] = Field(None, description="Fastest inference (ms)")
    max_latency_ms: Optional[float] = Field(None, description="Slowest inference (ms)")
    default_rate: Optional[float] = Field(None, description="Fraction of predictions flagged as default")
    risk_counts: Dict[str, int] = Field(default_factory=dict, description="Predictions per risk category")
    predictions: List[Dict[str, Any]] = Field(default_factory=list, description="Recent prediction records")


# ============================================================================
# Model Loading
# ============================================================================


def load_model(model_path: str) -> tuple:
    """
    Load the trained model, preprocessor and metadata from disk.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        tuple: (model, feature_names, model_metadata)
    """
    global model, preprocessor, feature_names, model_version, mlflow_run_id, model_metadata
    
    path = Path(model_path)
    if not path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model_data = joblib.load(path)
        
        if isinstance(model_data, dict):
            model = model_data.get('model')
            preprocessor = model_data.get('preprocessor')
            feature_names = model_data.get('feature_names')
            model_version = model_data.get('model_version', '1.0.0')
            mlflow_run_id = model_data.get('mlflow_run_id')
            model_metadata = {
                'model_type': model_data.get('model_type'),
                'best_threshold': model_data.get('best_threshold'),
                'best_f1': model_data.get('best_f1'),
                'dataset_version': model_data.get('dataset_version'),
                'form_coverage_pct': model_data.get('form_coverage_pct'),
                'feature_set': model_data.get('feature_set'),
                'mlflow_run_id': mlflow_run_id,
            }
        else:
            model = model_data
            preprocessor = None
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_.tolist()
            else:
                feature_names = None
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model version: {model_version}")
        if preprocessor is None:
            logger.warning("No preprocessor in artifact: raw inputs cannot be transformed (legacy artifact?)")
        else:
            logger.info(f"Preprocessor loaded: {len(preprocessor.feature_names_)} features")
        if model_metadata:
            logger.info(f"Model type: {model_metadata.get('model_type', 'unknown')}")
        if feature_names:
            logger.info(f"Feature count: {len(feature_names)}")
        if mlflow_run_id:
            logger.info(f"MLflow run ID: {mlflow_run_id}")
        
        return model, feature_names, model_metadata
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def prepare_input_data(input_data: PredictionInput) -> pd.DataFrame:
    """
    Build the model feature frame from a raw application input.
    
    Missing fields are filled with the training median/mode, then the
    preprocessor fitted at training time (polynomial features, domain
    ratios, label/one-hot encoding) is applied.
    
    Args:
        input_data: Validated raw input data
        
    Returns:
        pd.DataFrame: Single row DataFrame with the model features
    """
    if preprocessor is None:
        raise RuntimeError("Preprocessor not loaded: cannot transform raw inputs")
    
    values = input_data.model_dump()
    filled = {}
    for f in RAW_FIELDS:
        v = values.get(f["name"])
        filled[f["name"]] = f["default"] if v is None else v
    
    raw_df = pd.DataFrame([filled])
    return preprocessor.transform(raw_df)


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="Credit Scoring API with MLflow",
    description="""
    # Credit Scoring Prediction API with MLflow Tracking

    Predicts loan default probability from raw Home Credit application data.
    Inputs are transformed by the preprocessing pipeline fitted at training
    time (see api/ml_pipeline.py and api/train_model.py), so the
    /predict payload uses the raw application columns. Every prediction is
    tracked with MLflow for monitoring and analysis.

    ## Features
    - Dynamic input schema generated from the training data (see /schema)
    - Missing fields filled with training median/mode
    - Preprocessor + model loaded once at startup
    - MLflow tracking for all predictions
    - Automatic OpenAPI documentation
    """,
    version="1.0.0",
    contact={"name": "MLOps Team", "email": "mlops@example.com"},
    license_info={"name": "MIT"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

request_counter = 0


@app.middleware("http")
async def error_rate_middleware(request: Request, call_next):
    """Track request counts and error counts for /monitoring.

    FastAPI validation errors (422) are raised before the endpoint runs,
    but the middleware still sees the final response status, so they are
    counted as 4xx. Unhandled exceptions re-raise through here and count
    as 5xx.

    Each served request is also appended as a JSON line to REQUESTS_LOG_FILE
    (timestamp, method, path, status, duration) so the error rate and the
    request latency can be analyzed a posteriori.
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
        record_request(response.status_code)
        record_request_event(request.method, request.url.path,
                             response.status_code,
                             (time.perf_counter() - start) * 1000)
        return response
    except Exception:
        record_request(None)
        record_request_event(request.method, request.url.path, None,
                             (time.perf_counter() - start) * 1000)
        raise


@app.on_event("startup")
async def startup_event():
    """Load model when application starts"""
    global model, preprocessor, feature_names, model_version, mlflow_run_id, model_metadata
    
    logger.info("Starting API...")
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Raw input schema: {len(RAW_FIELDS)} fields from {SCHEMA_PATH}")
    
    try:
        model, feature_names, model_metadata = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        load_onnx_session()
        warm_up_inference()
        logger.info("API is ready")
    except FileNotFoundError:
        logger.warning(f"Model not found at {MODEL_PATH}")
        logger.warning("API started but predictions will fail")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", tags=["general"])
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model": "/model",
        "schema": "/schema",
        "predict": "/predict"
    }


@app.get("/schema", tags=["general"])
async def get_schema():
    """Raw input schema: field types, groups, defaults and categorical choices.

    Generated at training time (api/train_model.py) from the raw Home Credit
    columns. The Streamlit dashboard uses it to build its form dynamically.
    Numeric fields also carry the business ranges enforced by /predict
    (NUMERIC_CONSTRAINTS, as "min"/"max") so clients can validate before
    submitting instead of discovering a 422 after the fact.
    """
    fields = []
    for f in SCHEMA.get("fields", []):
        constraints = NUMERIC_CONSTRAINTS.get(f["name"], {})
        extra = {}
        if "ge" in constraints:
            extra["min"] = constraints["ge"]
        if "le" in constraints:
            extra["max"] = constraints["le"]
        fields.append({**f, **extra})
    return {**SCHEMA, "fields": fields}


@app.get("/health", response_model=HealthCheckOutput, tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    global request_counter
    request_counter += 1
    
    status_value = "healthy" if model is not None else "degraded"
    
    return HealthCheckOutput(
        status=status_value,
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        model_version=model_version if model else None,
        model_type=model_metadata.get('model_type') if model_metadata else None,
        mlflow_run_id=mlflow_run_id,
        feature_count=len(feature_names) if feature_names else None
    )


@app.get("/model", tags=["monitoring"])
async def get_model_info():
    """Get model metadata"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "preprocessor_loaded": preprocessor is not None,
        "inference_engine": "onnx" if onnx_session is not None else "lightgbm",
        "model_version": model_version,
        "model_type": model_metadata.get('model_type'),
        "feature_count": len(feature_names) if feature_names else 0,
        "features": feature_names,
        "raw_input_fields": len(RAW_FIELDS),
        "feature_importances": get_feature_importances(),
        # Share of the model's gain importance carried by the simplified
        # dashboard form fields. From the artifact metadata when present
        # (logged at training time), computed live from the loaded model
        # otherwise (e.g. artifacts trained before this field existed).
        "form_coverage_pct": (
            model_metadata.get("form_coverage_pct")
            if model_metadata and model_metadata.get("form_coverage_pct") is not None
            else compute_form_coverage(model, feature_names, [f["name"] for f in RAW_FIELDS])
        ),
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment": MLFLOW_EXPERIMENT,
            "run_id": mlflow_run_id,
        },
        "training_metadata": model_metadata
    }


@app.get("/monitoring", response_model=MonitoringOutput, tags=["monitoring"])
async def get_monitoring():
    """Live serving metrics: score distribution, inference latency, risk categories.

    Computed from the in-memory buffer of recent predictions. The buffer is
    reset on restart/redeploy (no persistent disk on the Render free plan);
    persistent tracking is available through MLflow.
    """
    with _prediction_lock:
        records = list(_recent_predictions)

    with _prediction_lock:
        stats = dict(_request_stats)

    def _error_rate(s: Dict[str, int]) -> Optional[float]:
        if not s["total"]:
            return None
        return round((s["errors_4xx"] + s["errors_5xx"]) / s["total"], 4)

    if not records:
        return MonitoringOutput(
            total_predictions=0,
            buffer_capacity=MONITORING_BUFFER_SIZE,
            total_requests=stats["total"],
            errors_4xx=stats["errors_4xx"],
            errors_5xx=stats["errors_5xx"],
            error_rate=_error_rate(stats),
        )

    latencies = sorted(r["processing_time_ms"] for r in records)
    risk_counts: Dict[str, int] = {}
    for r in records:
        risk_counts[r["risk_category"]] = risk_counts.get(r["risk_category"], 0) + 1
    n_defaults = sum(1 for r in records if r["prediction"] == 1)

    return MonitoringOutput(
        total_predictions=len(records),
        buffer_capacity=MONITORING_BUFFER_SIZE,
        total_requests=stats["total"],
        errors_4xx=stats["errors_4xx"],
        errors_5xx=stats["errors_5xx"],
        error_rate=_error_rate(stats),
        avg_latency_ms=round(sum(latencies) / len(latencies), 2),
        p95_latency_ms=round(_percentile(latencies, 95), 2),
        min_latency_ms=round(latencies[0], 2),
        max_latency_ms=round(latencies[-1], 2),
        default_rate=round(n_defaults / len(records), 4),
        risk_counts=risk_counts,
        predictions=records,
    )


@app.post("/predict", 
          response_model=PredictionOutput,
          tags=["prediction"],
          responses={400: {"description": "Invalid input"}, 422: {"description": "Validation error"}, 500: {"description": "Server error"}})
async def predict(input_data: PredictionInput):
    """
    Make a credit scoring prediction from a raw Home Credit application input.

    All fields are optional: omitted fields are filled with the training
    median/mode. Inputs are transformed by the preprocessor fitted at
    training time, and every prediction is logged to MLflow.
    """
    global request_counter
    request_counter += 1
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if preprocessor is None:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded (model artifact too old, retrain with api/train_model.py)")
    
    start_time = time.time()
    
    try:
        input_df = prepare_input_data(input_data)
        
        if hasattr(model, 'predict_proba'):
            # Single inference pass: the class is derived from the
            # probability (> 0.5, LightGBM's own decision rule) instead of a
            # second model.predict call (x2 inference cost, PERF_STUDY.md).
            probability = predict_probability(input_df)
            prediction = 1 if probability > 0.5 else 0
        elif hasattr(model, 'predict'):
            prediction = model.predict(input_df)[0]
            if isinstance(prediction, (float, np.floating)):
                probability = float(prediction)
                threshold = float(model_metadata.get('best_threshold') or 0.5) if model_metadata else 0.5
                prediction = 1 if probability > threshold else 0
            else:
                prediction = int(prediction)
                probability = 0.95 if prediction == 1 else 0.05
        else:
            raise ValueError("Model has no predict method")
        
        threshold = float(model_metadata.get('best_threshold') or 0.5) if model_metadata else 0.5
        
        if probability < threshold * 0.33:
            risk_category = "LOW_RISK"
        elif probability < threshold * 0.67:
            risk_category = "MEDIUM_RISK"
        else:
            risk_category = "HIGH_RISK"
        
        processing_time = time.time() - start_time
        
        # Record for the in-memory monitoring buffer (GET /monitoring)
        record_prediction(
            probability=probability,
            prediction=int(prediction),
            risk_category=risk_category,
            processing_time_ms=processing_time * 1000,
        )
        
        # Log to MLflow (off the request critical path by default)
        dispatch_prediction_logging(
            input_data=input_data.model_dump(),
            prediction=int(prediction),
            probability=probability,
            model_version=model_version,
            risk_category=risk_category,
            processing_time=processing_time
        )
        
        logger.info(f"Prediction #{request_counter} - {risk_category} (prob: {probability:.4f})")
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=round(probability, 4),
            risk_category=risk_category,
            model_version=model_version,
            model_type=model_metadata.get('model_type') if model_metadata else None,
            mlflow_run_id=mlflow_run_id,
            processing_time_ms=round(processing_time * 1000, 2)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Get API metrics"""
    global request_counter
    return {
        "requests": {"total": request_counter},
        "model": {
            "loaded": model is not None,
            "version": model_version if model else None,
            "type": model_metadata.get('model_type') if model_metadata else None
        }
    }
