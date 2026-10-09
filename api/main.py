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
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

from api.ml_pipeline import setup_mlflow_experiment


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Load environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "api/models/credit_scoring_model.pkl")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", str(Path(__file__).parent / "schema" / "raw_columns.json"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///tracking/mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "credit_scoring_api")

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


# ============================================================================
# MLflow Configuration
# ============================================================================


# Input fields logged as MLflow params (the full input is logged as a JSON artifact)
KEY_PARAMS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CODE_GENDER", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
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

def _build_prediction_input(fields: List[Dict[str, Any]]):
    """Build the Pydantic input model dynamically from the raw schema.

    All fields are optional: any field omitted in a request is filled with
    the training median (numeric) or mode (categorical) before prediction.
    """
    attrs: Dict[str, Any] = {}
    for f in fields:
        if f["type"] == "categorical":
            attrs[f["name"]] = (Optional[str], Field(default=None, description=f["description"]))
        else:
            attrs[f["name"]] = (Optional[float], Field(default=None, description=f["description"]))
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


@app.on_event("startup")
async def startup_event():
    """Load model when application starts"""
    global model, preprocessor, feature_names, model_version, mlflow_run_id, model_metadata
    
    logger.info("Starting API...")
    setup_mlflow_experiment(MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Raw input schema: {len(RAW_FIELDS)} fields from {SCHEMA_PATH}")
    
    try:
        model, feature_names, model_metadata = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
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
    """
    return SCHEMA


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
        "model_version": model_version,
        "model_type": model_metadata.get('model_type'),
        "feature_count": len(feature_names) if feature_names else 0,
        "features": feature_names,
        "raw_input_fields": len(RAW_FIELDS),
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment": MLFLOW_EXPERIMENT,
            "run_id": mlflow_run_id,
        },
        "training_metadata": model_metadata
    }


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
            proba = model.predict_proba(input_df)[0]
            prediction = model.predict(input_df)[0]
            probability = float(proba[1])
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
        
        # Log to MLflow
        log_prediction_to_mlflow(
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
