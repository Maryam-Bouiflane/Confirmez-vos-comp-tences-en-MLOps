"""
Unit and integration tests for the FastAPI application and its preprocessor.

These tests cover:
1. The CreditPreprocessor (train/serve parity on synthetic data)
2. Input handling (empty payload, wrong types, extra fields)
3. API endpoints (health check, schema, prediction)
4. Error handling (model/preprocessor not loaded)
5. Model loading (with and without preprocessor)
"""

import os
import json
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Set environment variable for model path
os.environ.setdefault("MODEL_PATH", "api/models/credit_scoring_model.pkl")

# Import after setting environment variable
from api.main import app, load_model
from api.ml_pipeline import (
    CreditPreprocessor,
    add_polynomial_features,
    add_domain_features,
    encode_categoricals,
    _sanitize_feature_names,
    compute_form_coverage,
)

# Create test client
client = TestClient(app)


# ============================================================================
# Fixtures and helpers
# ============================================================================

def make_synthetic_frame(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Build a small synthetic application frame with the Home Credit shape."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        'SK_ID_CURR': np.arange(100, 100 + n),
        'TARGET': rng.integers(0, 2, n),
        'EXT_SOURCE_1': rng.normal(0.5, 0.2, n),
        'EXT_SOURCE_2': rng.normal(0.4, 0.2, n),
        'EXT_SOURCE_3': rng.normal(0.5, 0.3, n),
        'DAYS_BIRTH': -rng.integers(8000, 25000, n),
        'AMT_CREDIT': rng.uniform(1e5, 2e6, n),
        'AMT_INCOME_TOTAL': rng.uniform(2e4, 5e5, n),
        'AMT_ANNUITY': rng.uniform(1e4, 1e5, n),
        'DAYS_EMPLOYED': -rng.integers(100, 5000, n),
        'FLAG_OWN_CAR': rng.choice(['Y', 'N'], n),          # binary -> label encoded
        'NAME_EDUCATION_TYPE': rng.choice(                  # multi -> one-hot
            ['Secondary', 'Higher', 'Incomplete', 'Lower'], n),
        'SOME_NUM': rng.normal(0, 1, n),
    })
    frame.loc[frame.index[3], 'EXT_SOURCE_1'] = np.nan  # test median imputation
    return frame


@pytest.fixture(scope="module")
def synthetic_preprocessor():
    """CreditPreprocessor fitted on the synthetic frame."""
    return CreditPreprocessor().fit(make_synthetic_frame())


def mock_model(probability: float = 0.8, prediction: int = 1) -> Mock:
    """Mock sklearn-like model with predict_proba/predict."""
    model = Mock()
    model.predict_proba = Mock(return_value=np.array([[1 - probability, probability]]))
    model.predict = Mock(return_value=np.array([prediction]))
    return model


class StubModel:
    """Minimal picklable model (joblib-compatible), for artifact tests."""

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])

    def predict(self, X):
        return np.array([1])


# ============================================================================
# CreditPreprocessor tests
# ============================================================================

class TestCreditPreprocessor:
    """Tests for the train/serve preprocessing parity."""

    def test_feature_order_matches_training_chain(self, synthetic_preprocessor):
        """feature_names_ must equal the sanitized training chain output."""
        train = make_synthetic_frame()
        ref_poly, ref_poly_te = add_polynomial_features(train, train)
        ref_fe, ref_fe_te = add_domain_features(ref_poly, ref_poly_te)
        ref_enc, _ = encode_categoricals(ref_fe, ref_fe_te)
        expected = _sanitize_feature_names(
            [c for c in ref_enc.columns if c not in ('SK_ID_CURR', 'TARGET')]
        )
        assert synthetic_preprocessor.feature_names_ == expected

    def test_transform_reproduces_training_values(self, synthetic_preprocessor):
        """transform() must return the same values as the training chain."""
        train = make_synthetic_frame()
        ref_poly, ref_poly_te = add_polynomial_features(train, train)
        ref_fe, ref_fe_te = add_domain_features(ref_poly, ref_poly_te)
        ref_enc, _ = encode_categoricals(ref_fe, ref_fe_te)
        ref_cols = [c for c in ref_enc.columns if c not in ('SK_ID_CURR', 'TARGET')]
        ref = ref_enc[ref_cols].apply(pd.to_numeric, errors='coerce').astype(float)
        ref.columns = _sanitize_feature_names(ref.columns)

        got = synthetic_preprocessor.transform(
            train.drop(columns=['SK_ID_CURR', 'TARGET'])
        )
        assert list(got.columns) == list(ref.columns)
        for col in ref.columns:
            expected_vals = np.nan_to_num(ref[col].to_numpy())
            assert np.allclose(expected_vals, got[col].to_numpy(), atol=1e-9)

    def test_transform_single_row_shape(self, synthetic_preprocessor):
        """A single partial row must produce one row with all features."""
        row = make_synthetic_frame(n=5).drop(
            columns=['SK_ID_CURR', 'TARGET', 'SOME_NUM']
        ).iloc[[0]]
        out = synthetic_preprocessor.transform(row)
        assert out.shape == (1, len(synthetic_preprocessor.feature_names_))

    def test_transform_unseen_category_is_neutral(self, synthetic_preprocessor):
        """Unseen one-hot and label categories map to neutral values."""
        row = make_synthetic_frame(n=5).drop(columns=['SK_ID_CURR', 'TARGET']).iloc[[0]]
        row['NAME_EDUCATION_TYPE'] = 'UNSEEN_VALUE'
        row['FLAG_OWN_CAR'] = 'MAYBE'
        out = synthetic_preprocessor.transform(row)
        # No exception, correct shape, unseen label -> -1
        assert out.shape[0] == 1
        assert out['FLAG_OWN_CAR'].iloc[0] == -1

    def test_transform_missing_column_filled_with_zero(self, synthetic_preprocessor):
        """A raw column absent from the input must be filled with 0."""
        row = make_synthetic_frame(n=5).drop(
            columns=['SK_ID_CURR', 'TARGET', 'SOME_NUM']
        ).iloc[[0]]
        out = synthetic_preprocessor.transform(row)
        assert out['SOME_NUM'].iloc[0] == 0.0

    def test_fit_requires_target(self):
        """fit() must reject a frame without SK_ID_CURR/TARGET."""
        with pytest.raises(ValueError):
            CreditPreprocessor().fit(pd.DataFrame({'A': [1, 2]}))

    def test_transform_before_fit_raises(self):
        """transform() on an unfitted preprocessor must raise."""
        with pytest.raises(RuntimeError):
            CreditPreprocessor().transform(pd.DataFrame({'A': [1]}))


# ============================================================================
# Schema and health endpoints
# ============================================================================

class TestSchemaEndpoint:
    """Tests for the /schema endpoint."""

    def test_schema_returns_200(self):
        response = client.get("/schema")
        assert response.status_code == 200

    def test_schema_structure(self):
        data = client.get("/schema").json()
        assert "fields" in data
        assert data["n_fields"] == len(data["fields"])
        for field in data["fields"]:
            assert field["name"] not in ("SK_ID_CURR", "TARGET")
            assert field["type"] in ("numeric", "categorical")
            assert "group" in field
            assert "default" in field

    def test_schema_exposes_business_ranges(self):
        """Numeric fields carry the /predict constraints as min/max."""
        by_name = {f["name"]: f for f in client.get("/schema").json()["fields"]}
        assert by_name["EXT_SOURCE_1"]["min"] == 0
        assert by_name["EXT_SOURCE_1"]["max"] == 1
        assert by_name["DAYS_BIRTH"]["max"] == -6570
        assert by_name["AMT_INCOME_TOTAL"]["min"] == 0
        assert "max" not in by_name["AMT_INCOME_TOTAL"]
        # One-sided constraint: min only, no upper bound
        assert by_name["AMT_CREDIT"]["min"] == 0
        assert "max" not in by_name["AMT_CREDIT"]
        # Categorical fields carry no bounds
        assert "min" not in by_name["CODE_GENDER"]
        assert "max" not in by_name["NAME_EDUCATION_TYPE"]


class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_health_check_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data
        assert "model_version" in data

    def test_health_check_degraded_without_model(self):
        """Without a model file, health reports degraded."""
        data = client.get("/health").json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


# ============================================================================
# Prediction endpoint
# ============================================================================

class TestPrediction:
    """Tests for the prediction endpoint."""

    def test_empty_payload_fills_defaults(self, synthetic_preprocessor):
        """All fields optional: {} must be filled with schema defaults."""
        with patch('api.main.model', mock_model(0.2, 0)):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                response = client.post("/predict", json={})
                assert response.status_code == 200
                data = response.json()
                assert data["prediction"] == 0
                assert "probability" in data
                assert "risk_category" in data

    def test_partial_payload_returns_200(self, synthetic_preprocessor):
        """A few raw fields must be enough to predict."""
        payload = {
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 500000.0,
            "DAYS_BIRTH": -12000,
            "EXT_SOURCE_1": 0.5,
            "CODE_GENDER": "M",
        }
        with patch('api.main.model', mock_model(0.8, 1)):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                response = client.post("/predict", json=payload)
                assert response.status_code == 200

    def test_wrong_type_returns_422(self, synthetic_preprocessor):
        """A string in a numeric field must be rejected by Pydantic."""
        with patch('api.main.model', mock_model()):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                response = client.post("/predict", json={"AMT_CREDIT": "not-a-number"})
                assert response.status_code == 422

    def test_extra_fields_ignored(self, synthetic_preprocessor):
        """Unknown fields must be ignored, not cause an error."""
        payload = {"AMT_CREDIT": 500000.0, "unknown_field": "ignored"}
        with patch('api.main.model', mock_model(0.5, 0)):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                response = client.post("/predict", json=payload)
                assert response.status_code == 200

    def test_prediction_output_structure(self, synthetic_preprocessor):
        with patch('api.main.model', mock_model(0.8, 1)):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                with patch('api.main.model_version', "1.0.0"):
                    data = client.post("/predict", json={}).json()
                    assert isinstance(data["prediction"], int)
                    assert isinstance(data["probability"], float)
                    assert isinstance(data["risk_category"], str)
                    assert isinstance(data["model_version"], str)

    def test_risk_category_classification(self, synthetic_preprocessor):
        """Risk categories must follow the threshold-based bands."""
        for probability, expected in [(0.10, "LOW_RISK"), (0.20, "MEDIUM_RISK"),
                                       (0.90, "HIGH_RISK")]:
            with patch('api.main.model', mock_model(probability, 1)):
                with patch('api.main.preprocessor', synthetic_preprocessor):
                    with patch('api.main.model_metadata', {'best_threshold': 0.5}):
                        data = client.post("/predict", json={}).json()
                        assert data["risk_category"] == expected


# ============================================================================
# Model loading
# ============================================================================

class TestInputRangeValidation:
    """Business range constraints on the raw input (422 on out-of-range)."""

    @pytest.mark.parametrize("field,value", [
        ("EXT_SOURCE_2", 5.0),           # normalized score must be in [0, 1]
        ("EXT_SOURCE_3", -0.1),
        ("DAYS_BIRTH", 100),             # positive => negative age
        ("DAYS_BIRTH", -70000),          # older than 110 years
        ("AMT_INCOME_TOTAL", -500.0),    # negative income
        ("CNT_CHILDREN", -1),
        ("AMT_CREDIT", -1.0),
        ("REGION_RATING_CLIENT", 7),     # rating only takes values 1-3
    ])
    def test_out_of_range_returns_422(self, synthetic_preprocessor, field, value):
        """Values outside the expected business range must be rejected."""
        with patch('api.main.model', mock_model()):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                response = client.post("/predict", json={field: value})
                assert response.status_code == 422

    def test_boundary_values_accepted(self, synthetic_preprocessor):
        """Boundary values of the ranges are valid inputs (inclusive)."""
        payload = {"DAYS_BIRTH": -6570, "EXT_SOURCE_1": 1.0, "AMT_INCOME_TOTAL": 0.0}
        with patch('api.main.model', mock_model(0.5, 0)):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                response = client.post("/predict", json=payload)
                assert response.status_code == 200


class TestModelLoading:
    """Tests for model loading functionality."""

    def test_load_nonexistent_model_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.pkl")

    def test_load_artifact_with_preprocessor(self, synthetic_preprocessor):
        """A current artifact (dict with preprocessor) loads both."""
        import joblib
        model_data = {
            'model': StubModel(),
            'preprocessor': synthetic_preprocessor,
            'feature_names': synthetic_preprocessor.feature_names_,
            'model_version': '1.0.0',
            'model_type': 'lightgbm',
        }
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            joblib.dump(model_data, f)
            temp_path = f.name

        try:
            with patch('api.main.model', None), \
                 patch('api.main.preprocessor', None):
                model, feature_names, _ = load_model(temp_path)
                assert model is not None
                assert feature_names == synthetic_preprocessor.feature_names_
        finally:
            os.unlink(temp_path)

    def test_load_legacy_artifact_warns(self):
        """An artifact without preprocessor keeps preprocessor as None."""
        import joblib
        model_data = {
            'model': StubModel(),
            'feature_names': ['f1', 'f2'],
            'model_version': '1.0.0',
        }
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            joblib.dump(model_data, f)
            temp_path = f.name

        try:
            with patch('api.main.model', None), \
                 patch('api.main.preprocessor', 'sentinel'):
                load_model(temp_path)
                from api import main as api_main
                assert api_main.preprocessor is None
        finally:
            os.unlink(temp_path)


# ============================================================================
# Root endpoint and error handling
# ============================================================================

class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_endpoint(self):
        data = client.get("/").json()
        assert data["message"] == "Credit Scoring API"
        assert data["version"] == "1.0.0"
        assert "schema" in data
        assert "health" in data


class TestErrorHandling:
    """Tests for error handling."""

    def test_model_not_loaded_error(self):
        with patch('api.main.model', None):
            response = client.post("/predict", json={})
            assert response.status_code == 500
            assert "Model not loaded" in response.text

    def test_preprocessor_not_loaded_error(self):
        with patch('api.main.model', mock_model()):
            with patch('api.main.preprocessor', None):
                response = client.post("/predict", json={})
                assert response.status_code == 500
                assert "Preprocessor not loaded" in response.text

    def test_predict_with_probability_only_model(self, synthetic_preprocessor):
        """A model without predict_proba falls back to predict + threshold."""
        model = Mock(spec=['predict'])
        model.predict = Mock(return_value=np.array([1]))
        with patch('api.main.model', model):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                with patch('api.main.model_metadata', {'best_threshold': 0.5}):
                    data = client.post("/predict", json={"AMT_CREDIT": 1.0}).json()
                    assert data["prediction"] == 1

    def test_model_endpoint_500_without_model(self):
        with patch('api.main.model', None):
            assert client.get("/model").status_code == 500


class TestStartupAndMonitoring:
    """Tests for the startup event and monitoring endpoints."""

    def test_startup_degraded_without_model_file(self):
        """Entering the app context with an unfindable model -> degraded.

        The model artifact is committed to the repo (the Docker image bakes
        it in), so the missing-file scenario is simulated by patching
        MODEL_PATH to a nonexistent path and clearing any loaded model.
        """
        with patch("api.main.MODEL_PATH", "/nonexistent/model.pkl"), \
             patch("api.main.model", None), \
             patch("api.main.preprocessor", None):
            with TestClient(app) as c:
                data = c.get("/health").json()
                assert data["status"] == "degraded"
                assert data["model_loaded"] is False

    def test_metrics_endpoint(self):
        data = client.get("/metrics").json()
        assert "requests" in data
        assert "model" in data


class TestMonitoringEndpoint:
    """Tests for the live monitoring endpoint (/monitoring)."""

    def test_monitoring_structure(self):
        data = client.get("/monitoring").json()
        assert data["buffer_capacity"] == 500
        assert isinstance(data["predictions"], list)
        assert isinstance(data["risk_counts"], dict)

    def test_monitoring_records_predictions(self, synthetic_preprocessor):
        with patch('api.main.model', mock_model(0.8, 1)):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                before = client.get("/monitoring").json()["total_predictions"]
                response = client.post("/predict", json={})
                assert response.status_code == 200

                data = client.get("/monitoring").json()
                assert data["total_predictions"] == before + 1
                last = data["predictions"][-1]
                assert last["probability"] == 0.8
                assert last["prediction"] == 1
                assert last["risk_category"] in ("LOW_RISK", "MEDIUM_RISK", "HIGH_RISK")
                assert "processing_time_ms" in last
                assert "timestamp" in last
                assert data["risk_counts"][last["risk_category"]] >= 1
                assert data["avg_latency_ms"] is not None
                assert data["min_latency_ms"] <= data["p95_latency_ms"] <= data["max_latency_ms"]

    def test_model_exposes_feature_importances_key(self):
        with patch('api.main.model', mock_model(0.8, 1)):
            with patch('api.main.feature_names', ["f1", "f2"]):
                data = client.get("/model").json()
                assert "feature_importances" in data
                assert isinstance(data["feature_importances"], list)


class TestErrorRateTracking:
    """The /monitoring endpoint must expose request counts and error rates."""

    def _reset(self):
        from api.main import _request_stats
        _request_stats.update(total=0, errors_4xx=0, errors_5xx=0)

    def test_error_rate_counts_4xx(self, synthetic_preprocessor):
        """A Pydantic validation error (422) must show up in the error rate."""
        self._reset()
        with patch('api.main.model', mock_model()):
            with patch('api.main.preprocessor', synthetic_preprocessor):
                response = client.post("/predict", json={"EXT_SOURCE_2": 5.0})
                assert response.status_code == 422
        data = client.get("/monitoring").json()
        assert data["total_requests"] >= 1
        assert data["errors_4xx"] >= 1
        assert data["error_rate"] is not None and data["error_rate"] > 0

    def test_error_rate_counts_5xx(self):
        """A server error (model not loaded -> 500) must show up as 5xx."""
        self._reset()
        with patch('api.main.model', None):
            response = client.post("/predict", json={})
            assert response.status_code == 500
        data = client.get("/monitoring").json()
        assert data["errors_5xx"] >= 1
        assert data["error_rate"] is not None and data["error_rate"] > 0

    def test_error_rate_none_when_no_traffic(self):
        self._reset()
        # The middleware counts a request once it completes, so the current
        # /monitoring call appears in the NEXT snapshot: read twice.
        client.get("/monitoring")
        data = client.get("/monitoring").json()
        assert data["total_requests"] == 1
        assert data["errors_4xx"] == 0
        assert data["errors_5xx"] == 0
        assert data["error_rate"] == 0.0

    def test_requests_log_writes_json_lines(self, tmp_path, monkeypatch):
        """Each served request is appended as a JSON line (persistent log)."""
        import api.main as main_module
        log_path = str(tmp_path / "api_requests.jsonl")
        monkeypatch.setattr(main_module, "REQUESTS_LOG_FILE", log_path)
        client.get("/health")
        with open(log_path, encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        assert len(events) >= 1
        event = events[-1]
        assert event["method"] == "GET"
        assert event["path"] == "/health"
        assert event["status_code"] == 200
        assert event["duration_ms"] >= 0
        assert "timestamp" in event

    def test_requests_log_disabled_on_empty_path(self, monkeypatch):
        """An empty REQUESTS_LOG_FILE turns the persistent log off."""
        import api.main as main_module
        monkeypatch.setattr(main_module, "REQUESTS_LOG_FILE", "")
        # Must not raise: the request still completes
        response = client.get("/health")
        assert response.status_code == 200


class TestFormCoverage:
    """Tests for the form-fields importance coverage computation."""

    def test_coverage_fractional_attribution(self):
        class FakeModel:
            feature_importances_ = np.array([10.0, 5.0, 5.0])

        feats = ["EXT_SOURCE_1", "AMT_CREDIT", "FLAG_OWN_CAR_Y"]
        raw = ["EXT_SOURCE_1", "AMT_CREDIT", "FLAG_OWN_CAR"]
        # EXT_SOURCE_1 and AMT_CREDIT are in TOP_FORM_FIELDS (15/20 of the gain),
        # FLAG_OWN_CAR_Y (one-hot of FLAG_OWN_CAR, not in the top list) is not.
        cov = compute_form_coverage(FakeModel(), feats, raw)
        assert abs(cov - 75.0) < 0.1

    def test_coverage_without_importances_returns_zero(self):
        assert compute_form_coverage(object(), ["A"], ["A"]) == 0.0

    def test_model_endpoint_serves_coverage(self):
        with patch('api.main.model', mock_model(0.8, 1)):
            with patch('api.main.feature_names', ["f1", "f2"]):
                data = client.get("/model").json()
                assert "form_coverage_pct" in data
                assert data["form_coverage_pct"] is None or 0 <= data["form_coverage_pct"] <= 100


class TestOpenAPISchema:
    """Tests for OpenAPI schema generation."""

    def test_openapi_schema_generated(self):
        data = client.get("/openapi.json").json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Credit Scoring API with MLflow"

    def test_docs_available(self):
        assert client.get("/docs").status_code == 200


# ============================================================================
# Inference optimization (PERF_STUDY.md): single-pass prediction, ONNX engine,
# warm-up, async MLflow logging
# ============================================================================

class TestSinglePassPrediction:
    """/predict must derive the class from the probability (one inference),
    not call model.predict a second time."""

    def _predict(self, synthetic_preprocessor, proba_default):
        m = mock_model(probability=proba_default, prediction=99)  # predict() would say 99
        with patch('api.main.model', m), patch('api.main.preprocessor', synthetic_preprocessor):
            response = client.post("/predict", json={})
        assert response.status_code == 200
        return response.json(), m

    def test_prediction_follows_probability_not_model_predict(self, synthetic_preprocessor):
        data, m = self._predict(synthetic_preprocessor, proba_default=0.9)
        assert data["prediction"] == 1          # derived: 0.9 > 0.5
        m.predict.assert_not_called()           # single inference pass

    def test_below_threshold_prediction_is_zero(self, synthetic_preprocessor):
        data, _ = self._predict(synthetic_preprocessor, proba_default=0.3)
        assert data["prediction"] == 0


class TestOnnxEngine:
    """ONNX Runtime engine: load, parity with the LightGBM reference."""

    def _load_artifact(self):
        pytest.importorskip("onnxruntime")
        path = os.environ.get("MODEL_PATH", "api/models/credit_scoring_model.pkl")
        if not os.path.exists(path):
            pytest.skip("model artifact not available")
        import joblib
        return joblib.load(path)

    def test_onnx_matches_lightgbm_probabilities(self):
        import api.main as main_module
        artifact = self._load_artifact()
        onnx_path = main_module.ONNX_MODEL_PATH
        if not os.path.exists(onnx_path):
            pytest.skip("ONNX model not generated (run monitoring/optimize_model.py)")

        rng = np.random.default_rng(7)
        rows = []
        for _ in range(5):
            row = {}
            for f in main_module.RAW_FIELDS:
                name, default = f["name"], f["default"]
                if f["type"] == "categorical":
                    row[name] = default
                elif name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
                    row[name] = float(rng.uniform(0.1, 0.9))
                elif "AMT" in name:
                    row[name] = float(rng.uniform(1e4, 5e5))
                elif name.startswith("DAYS"):
                    row[name] = float(-rng.integers(1000, 20000))
                else:
                    row[name] = float(default)
            rows.append(row)

        saved = (main_module.model, main_module.preprocessor,
                 main_module.feature_names, main_module.onnx_session)
        try:
            main_module.model = artifact["model"]
            main_module.preprocessor = artifact["preprocessor"]
            main_module.feature_names = artifact["feature_names"]
            main_module.onnx_session = None
            main_module.load_onnx_session()
            if main_module.onnx_session is None:
                pytest.skip("ONNX session could not be created")
            for row in rows:
                X = artifact["preprocessor"].transform(pd.DataFrame([row]))
                p_onnx = main_module.predict_probability(X)
                p_skl = float(artifact["model"].predict_proba(X)[0][1])
                assert abs(p_onnx - p_skl) < 1e-4
        finally:
            (main_module.model, main_module.preprocessor,
             main_module.feature_names, main_module.onnx_session) = saved

    def test_fallback_when_onnx_missing(self, tmp_path, monkeypatch):
        """auto mode + no .onnx file -> engine stays LightGBM, no crash."""
        import api.main as main_module
        monkeypatch.setattr(main_module, "ONNX_MODEL_PATH", str(tmp_path / "missing.onnx"))
        saved_session = main_module.onnx_session
        main_module.onnx_session = None
        try:
            main_module.load_onnx_session()
            assert main_module.onnx_session is None
        finally:
            main_module.onnx_session = saved_session


class TestWarmUp:
    """Startup warm-up must run one prediction without raising."""

    def test_warm_up_runs_one_prediction(self, synthetic_preprocessor):
        import api.main as main_module
        m = mock_model(probability=0.4, prediction=0)
        with patch('api.main.model', m), patch('api.main.preprocessor', synthetic_preprocessor):
            main_module.warm_up_inference()  # must not raise
        m.predict_proba.assert_called()

    def test_warm_up_noop_without_model(self):
        import api.main as main_module
        with patch('api.main.model', None):
            main_module.warm_up_inference()  # must not raise


class TestMlflowLoggingModes:
    """MLFLOW_LOG_MODE: async (default) | sync | off."""

    def _payload(self):
        return {"EXT_SOURCE_2": 0.5}, 1, 0.42, "1.0.0", "LOW_RISK", 0.012

    def test_mode_off_skips_logging(self, monkeypatch):
        import api.main as main_module
        monkeypatch.setattr(main_module, "MLFLOW_LOG_MODE", "off")
        with patch.object(main_module, "log_prediction_to_mlflow") as m:
            main_module.dispatch_prediction_logging(*self._payload())
        m.assert_not_called()

    def test_mode_sync_logs_inline(self, monkeypatch):
        import api.main as main_module
        monkeypatch.setattr(main_module, "MLFLOW_LOG_MODE", "sync")
        with patch.object(main_module, "log_prediction_to_mlflow") as m:
            main_module.dispatch_prediction_logging(*self._payload())
        m.assert_called_once()

    def test_mode_async_submits_and_completes(self, monkeypatch):
        import api.main as main_module
        monkeypatch.setattr(main_module, "MLFLOW_LOG_MODE", "async")
        with patch.object(main_module, "log_prediction_to_mlflow") as m:
            main_module.dispatch_prediction_logging(*self._payload())
            if main_module._mlflow_executor is not None:
                main_module._mlflow_executor.shutdown(wait=True)
        m.assert_called_once()


    def test_redoc_available(self):
        assert client.get("/redoc").status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
