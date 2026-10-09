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
        """Entering the app context runs startup: no model file -> degraded."""
        with TestClient(app) as c:
            data = c.get("/health").json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False

    def test_metrics_endpoint(self):
        data = client.get("/metrics").json()
        assert "requests" in data
        assert "model" in data


class TestOpenAPISchema:
    """Tests for OpenAPI schema generation."""

    def test_openapi_schema_generated(self):
        data = client.get("/openapi.json").json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Credit Scoring API with MLflow"

    def test_docs_available(self):
        assert client.get("/docs").status_code == 200

    def test_redoc_available(self):
        assert client.get("/redoc").status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
