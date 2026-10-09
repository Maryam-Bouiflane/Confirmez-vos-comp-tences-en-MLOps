# Credit Scoring API - Deployment Guide

## 📋 Overview

This guide explains how to deploy the Credit Scoring API project with FastAPI, Docker, and CI/CD pipeline.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User / Browser                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│                    (Port: 8501)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP Requests
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│                    (Port: 8000)                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  • Model loaded at startup (singleton)                  ││
│  │  • Input validation with Pydantic                       ││
│  │  • Health check endpoint                                ││
│  │  • Prediction endpoint                                  ││
│  │  • Automatic documentation (Swagger/ReDoc)              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Pre-trained Model                         │
│              (api/models/credit_scoring_model.pkl)          │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
credit-scoring-api/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── ml_pipeline.py       # Feature engineering, preprocessor, training (K-Fold, Optuna)
│   ├── train_model.py       # Model training script (CLI)
│   ├── schema/
│   │   └── raw_columns.json # Raw input schema (generated at training time)
│   └── models/
│       └── credit_scoring_model.pkl  # Pre-trained model artifact
│
├── streamlit_app/
│   └── app.py               # Streamlit dashboard
│
├── tests/
│   └── test_api.py          # Unit and integration tests
│
├── data/raw/                # Home Credit CSVs (required for training)
│
├── tracking/                # MLflow database + artifacts, Optuna studies
│                            # (git-ignored, regenerated automatically)
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # GitHub Actions CI/CD pipeline
│
├── Dockerfile               # Docker configuration for API (multi-stage, non-root user)
├── Dockerfile.streamlit     # Docker configuration for Streamlit
├── docker-compose.yml       # Docker Compose for local development
│                            # (named volume api-tracking mounted on /app/tracking)
├── render.yaml              # Render deployment configuration (prod + staging)
├── pyproject.toml           # Python dependencies (uv) + mypy config
├── uv.lock                  # Lockfile (committed for reproducibility)
└── API_DEPLOYMENT_GUIDE.md  # This file
```

## ⚙️ Quick Start

### 1. Install Dependencies

```bash
uv sync                  # runtime dependencies
uv sync --all-extras     # including dev tools (pytest, mypy...)
```

### 2. Train the Model

```bash
# Train and save the model (full dataset)
uv run python -m api.train_model

# Quick run (subset of rows, as used in CI)
uv run python -m api.train_model --nrows 20000

# With Optuna hyperparameter tuning and Model Registry registration
uv run python -m api.train_model --tune --n-trials 50 --register
```

Main options: `--nrows`, `--model-type` (`lightgbm`, `logistic_regression`,
`random_forest`), `--tune`, `--n-trials`, `--n-folds`, `--experiment-name`,
`--run-name`, `--tracking-uri`, `--register`, `--output-path`.
Full list: `uv run python -m api.train_model --help`.

This will create a pre-trained model at `api/models/credit_scoring_model.pkl`.

### 3. Run the API Locally

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: http://localhost:8000

- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Predict**: POST http://localhost:8000/predict

### 4. Run the Streamlit Dashboard

```bash
uv run streamlit run streamlit_app/app.py --server.port=8501
```

The dashboard will be available at: http://localhost:8501

## 🐳 Docker Deployment

### Build the Docker Image

```bash
# Build the API image
docker build -t credit-scoring-api .

# Build the Streamlit image
docker build -t credit-scoring-api-streamlit -f Dockerfile.streamlit .
```

Note: build the images **after** training, as the Dockerfile copies
`api/models/credit_scoring_model.pkl` into the image.

### Run with Docker Compose

```bash
# Build and start both API and Streamlit
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The services will be available at:
- API: http://localhost:8000
- Streamlit: http://localhost:8501

## 🚀 GitHub Actions CI/CD

The CI/CD pipeline is configured in `.github/workflows/ci-cd.yml`. It automates:

1. **Test**: Run unit and integration tests
2. **Build**: Build Docker image
3. **Deploy to Staging**: Deploy to Render staging environment
4. **Deploy to Production**: Deploy to Render production environment

### Triggers

- **Push to main/develop**: Runs all jobs (test, build, deploy-staging, deploy-production)
- **Push to feature/* branches**: Runs test and build only
- **Pull Request**: Runs test only
- **Tag push (v*)**: Runs test, build, and deploy-production

### Secrets Required

Configure these secrets in your GitHub repository settings:

| Secret Name | Description |
|-------------|-------------|
| `RENDER_API_KEY` | Render.com API key for deployment |
| `RENDER_SERVICE_ID` | Staging service ID on Render |
| `RENDER_PROD_SERVICE_ID` | Production service ID on Render |

### View Pipeline

After pushing code, view the pipeline at:
`https://github.com/<your-username>/<your-repo>/actions`

## ☁️ Render Deployment

### Manual Deployment

1. **Push to GitHub**: Ensure your code is pushed to GitHub

2. **Connect to Render**:
   ```bash
   # Install Render CLI
   # See: https://render.com/docs/cli
   
   # Login
   render account login
   
   # Link repository
   render repo connect
   ```

3. **Deploy API Service**:
   ```bash
   # Create service from blueprint
   render blueprint deploy
   
   # Or for existing service
   render deploy <service-id>
   ```

4. **Update Streamlit API_URL**: After API is deployed, update the `API_URL` environment variable in the Streamlit service to point to your deployed API URL.

### Using render.yaml

The `render.yaml` file contains the configuration for both services:
- `credit-scoring-api` (FastAPI backend)
- `credit-scoring-dashboard` (Streamlit frontend)

Update the `API_URL` in the Streamlit service configuration after the API service is deployed.

### Render Service URLs

After deployment, your services will be available at:
- API: `https://credit-scoring-api-lnqc.onrender.com`
- Dashboard: `https://credit-scoring-dashboard-o0a7.onrender.com`

## 🧪 Testing

### Run All Tests

```bash
uv run pytest tests/ -v --cov=api.main
```

### Run Specific Tests

```bash
# Health check tests
uv run pytest tests/test_api.py::TestHealthCheck -v

# Input validation tests
uv run pytest tests/test_api.py::TestPredictionInputValidation -v

# Full API test file (unit + integration)
uv run pytest tests/test_api.py -v
```

### Test Coverage

```bash
# Generate coverage report
uv run pytest tests/ --cov=api.main --cov-report=html

# View coverage report (Windows)
start htmlcov/index.html
```

## 📝 API Documentation

The API provides automatic documentation via Swagger and ReDoc:

### Swagger UI

Available at: `/docs`

Provides interactive documentation where you can:
- View all endpoints
- Try out requests directly in the browser
- See request/response schemas

### ReDoc

Available at: `/redoc`

Provides a more compact, readable documentation format.

### OpenAPI Schema

Available at: `/openapi.json`

The raw OpenAPI specification in JSON format.

## 🎯 API Endpoints

### GET /

**Description**: API information

**Response**:
```json
{
  "message": "Credit Scoring API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "model": "/model",
  "schema": "/schema",
  "predict": "/predict"
}
```

### GET /health

**Description**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model_version": "1.0.0",
  "model_type": "lightgbm",
  "mlflow_run_id": "56d0ec879b984c2ead508a18b90c7324",
  "feature_count": 280
}
```

### POST /predict

**Description**: Make a credit scoring prediction

**Request Body**: the raw Home Credit application columns (120 fields, all
optional — omitted fields are filled with the training median/mode by the
preprocessor). The full list with types, groups and categorical choices is
served by `GET /schema`.

```json
{
  "AMT_INCOME_TOTAL": 202500.0,
  "AMT_CREDIT": 500000.0,
  "AMT_ANNUITY": 30000.0,
  "AMT_GOODS_PRICE": 450000.0,
  "DAYS_BIRTH": -12000,
  "DAYS_EMPLOYED": -1500,
  "EXT_SOURCE_1": 0.5,
  "EXT_SOURCE_2": 0.6,
  "EXT_SOURCE_3": 0.7,
  "CODE_GENDER": "M",
  "CNT_CHILDREN": 0,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_FAMILY_STATUS": "Married",
  "OCCUPATION_TYPE": "Laborers"
}
```

**Response**:
```json
{
  "prediction": 0,
  "probability": 0.3815,
  "risk_category": "MEDIUM_RISK",
  "model_version": "1.0.0",
  "model_type": "lightgbm",
  "mlflow_run_id": "56d0ec879b984c2ead508a18b90c7324",
  "processing_time_ms": 62.72
}
```

### Error Responses

**422 Unprocessable Entity**: Validation error (wrong type on a field)
```json
{
  "detail": [
    {
      "type": "float_parsing",
      "loc": ["body", "AMT_CREDIT"],
      "msg": "Input should be a valid number, unable to parse string as a number",
      "input": "five hundred thousand"
    }
  ]
}
```

**500 Internal Server Error**: Model not loaded or other server error
```json
{
  "detail": "Model not loaded. Please check API health."
}
```

## 🔒 Input Validation

The `PredictionInput` Pydantic model is generated **dynamically at startup**
from `api/schema/raw_columns.json`, itself generated at training time by
`api/train_model.py`. Validation rules:

| Rule | Detail |
|------|--------|
| Fields | The 120 raw Home Credit application columns (see `GET /schema`) |
| Required | None — all fields are optional |
| Numeric fields | Must parse as `float` (422 otherwise) |
| Categorical fields | Any string accepted; values outside the training choices get a neutral encoding |
| Missing fields | Filled with the training median (numeric) or mode (categorical) before prediction |

The request schema is therefore always in sync with what the model was
trained on: retraining regenerates the schema, and the API adapts on the
next start. The interactive field-by-field documentation is available in
the Swagger UI (`/docs`).

## 🎨 Risk Categories

The model classifies predictions into three risk categories, **relative to
the decision threshold optimized at training time** (`best_threshold`, stored
in the model artifact and visible on `/model` — 0.6627 for the current model):

| Category | Probability Range | Recommendation |
|----------|------------------|----------------|
| LOW_RISK | < threshold × 0.33 | Approve with standard terms |
| MEDIUM_RISK | threshold × 0.33 – 0.67 | Approve with caution |
| HIGH_RISK | ≥ threshold × 0.67 | Reject or require modifications |

## 📊 MLflow Tracking and Logging

All training runs and API predictions are tracked with MLflow for complete visibility.

### Starting MLflow UI

```bash
# Start MLflow web interface
mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db
```

Access the UI at: **http://localhost:5000**

### Viewing Training Runs

1. Open MLflow UI (http://localhost:5000)
2. Select the **`credit_scoring_api`** experiment
3. Click on any run to see details
4. Explore the tabs:
   - **Overview**: Summary of metrics, params, tags
   - **Metrics**: Charts for cv_train_auc, cv_valid_auc, best_f1, etc.
   - **Params**: Model hyperparameters, best_threshold, etc.
   - **Artifacts**: Feature importances, metrics, submissions
   - **Model**: Saved trained model

### Viewing API Prediction Logs

All API predictions are logged to MLflow with:
- **Run Name**: `api_prediction_YYYYMMDD_HHMMSS...`
- **Tags**: `source=api`, `predictor=credit_scoring`, `model_version`, `risk_category`
- **Params**: Key input features (income, credit amount, annuity, external scores, etc.); the full input is logged as a JSON artifact
- **Metrics**: prediction (0/1), probability, processing_time_ms

To view API predictions in the UI:
1. Filter by tag: `source=api`
2. Or search for runs starting with: `api_prediction_`

### Accessing MLflow Programmatically

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///tracking/mlflow.db")

# List experiments
client = mlflow.tracking.MlflowClient()
experiments = client.list_experiments()

# Get runs from specific experiment
exp = [e for e in experiments if e.name == "credit_scoring_api"][0]
runs = client.search_runs(experiment_ids=[exp.experiment_id])

# Get metrics from a run
run_id = runs[0].info.run_id
metrics = client.get_run(run_id).data.metrics
print(f"CV AUC: {metrics['cv_valid_auc']}")

# Load model for inference
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)
```

### Artifacts Structure

```
tracking/mlruns/
└── <experiment-id>/
    └── <run-id>/
        ├── artifacts/
        │   ├── feature_importances/      # Feature importance CSV
        │   │   └── feature_importances.csv
        │   ├── metrics/                   # CV metrics
        │   │   └── metrics.csv
        │   ├── optuna/                    # Optuna trials (if tuning)
        │   │   └── optuna_trials.csv
        │   └── submission/                # Predictions
        │       ├── submission_class.csv
        │       └── submission_proba.csv
        └── model/                        # Trained model
            └── MLmodel
```

Access artifacts directly:
```bash
# List artifacts
ls tracking/mlruns/<experiment-id>/<run-id>/artifacts/

# View metrics
cat tracking/mlruns/<experiment-id>/<run-id>/artifacts/metrics/metrics.csv

# View feature importances
cat tracking/mlruns/<experiment-id>/<run-id>/artifacts/feature_importances/feature_importances.csv
```

## 📊 Model Information

### Current Model

- **Type**: LightGBM (`LGBMClassifier`)
- **Version**: 1.0.0
- **Dataset**: Home Credit `application_train` (dataset version `v1_application_raw`)
- **Inputs**: 120 raw application columns (all optional, see `/schema`)
- **Engineered features**: ~280 (see below), produced by `CreditPreprocessor`
- **Decision threshold**: 0.6627, optimized for F1 on CV predictions (best CV F1: 0.313)
- **Target**: Binary classification (1 = loan default, 0 = no default)
- **MLflow run**: `56d0ec879b984c2ead508a18b90c7324` (also returned by `/model`)

### Feature Engineering (`api/ml_pipeline.py`)

The preprocessor fitted at training time (and re-applied identically at
serving time) performs:

1. **Missing-value imputation**: median (numeric) / mode (categorical)
2. **Polynomial features** (degree 3) on `EXT_SOURCE_1/2/3` and `DAYS_BIRTH`
3. **Domain features**: credit/income and annuity/income ratios, etc.
4. **Categorical encoding**: label encoding for binary columns, one-hot for the others

The full list of engineered features is available on `GET /model`
(`features` field).

### Performance Metrics

Tracked in MLflow for every training run (UI: http://localhost:5000):
`cv_train_auc`, `cv_valid_auc`, `best_f1`, `best_threshold`, plus feature
importances and the trained model as artifacts. See the MLflow section below.

## 🔧 Customization

### Retrain with Your Own Data or Model

The recommended path is always `api/train_model.py`, which regenerates the
**full artifact set** consistently (model + preprocessor + metadata + raw
schema):

```bash
# Different estimator
uv run python -m api.train_model --model-type random_forest

# With hyperparameter tuning
uv run python -m api.train_model --tune --n-trials 50
```

This overwrites `api/models/credit_scoring_model.pkl` and regenerates
`api/schema/raw_columns.json`. The API picks up both on the next start:
the Pydantic input model is rebuilt from the schema, so request validation
automatically follows whatever columns the new model was trained on.

Do **not** hand-craft the pickle: the artifact must embed the
`CreditPreprocessor` fitted at training time (train/serve parity), which
only `train_model.py` produces.

### Update Input Validation Rules

There is no hardcoded input class to edit: `PredictionInput` is generated
dynamically from `api/schema/raw_columns.json`. To change the accepted
fields, change the training input columns (`api/ml_pipeline.py`) and
retrain.

### Add New Endpoints

Add new endpoints to `api/main.py` following the FastAPI pattern:

```python
@app.get("/new-endpoint")
async def new_endpoint():
    return {"message": "New endpoint"}
```

## 🐛 Troubleshooting

### API not starting

**Issue**: `ModuleNotFoundError` for a package

**Solution**:
```bash
uv sync
```

### Model not loading

**Issue**: `FileNotFoundError: Model file not found`

**Solution**:
```bash
# Train the model first
uv run python -m api.train_model
```

### Validation errors

**Issue**: `422 Unprocessable Entity`

**Solution**: All input fields are optional — check that numeric fields
contain valid numbers and that categorical fields are strings (see
`GET /schema` for the expected types and choices).

### Docker build fails

**Issue**: Docker build error

**Solution**:
```bash
# Clean and rebuild
docker system prune -f
docker build -t credit-scoring-api .
```

### Port already in use

**Issue**: `Address already in use` / port 8000 or 8501 occupied

**Solution**:
```bash
# Linux / macOS
lsof -i :8000          # Find process ID
kill -9 <PID>          # Kill process

# Windows (PowerShell)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Render Documentation](https://render.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🎓 Best Practices Implemented

1. ✅ **Model Loading**: Model loaded once at startup, not per request
2. ✅ **Input Validation**: Comprehensive validation with Pydantic
3. ✅ **Error Handling**: Proper error handling and HTTP status codes
4. ✅ **Documentation**: Automatic API documentation with Swagger/ReDoc
5. ✅ **Testing**: Unit tests, integration tests, and test coverage
6. ✅ **Docker**: Multi-stage builds for smaller images
7. ✅ **Security**: Non-root user in Docker containers
8. ✅ **CI/CD**: Automated pipeline with GitHub Actions
9. ✅ **Health Checks**: Docker and API health checks
10. ✅ **Logging**: Structured logging for debugging

## 📞 Support

For issues or questions:

1. Check this deployment guide
2. Review the API logs
3. Verify all dependencies are installed
4. Ensure the model file exists
5. Check Docker is running (for containerized deployment)

## 📄 License

This project is licensed under the MIT License.
