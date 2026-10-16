"""
Credit Scoring Dashboard - Streamlit Application

This dashboard allows users to:
1. Input client information through a form
2. Send requests to the FastAPI backend
3. View prediction results and risk analysis
4. Explore model performance (if test data is available)

Usage:
    streamlit run streamlit_app/app.py

Environment Variables:
    API_URL: URL of the FastAPI backend (default: http://localhost:8000)
"""

import os
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

import streamlit as st


# Configuration
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
# API_URL can be a full URL (http://localhost:8000) or a bare hostname
# injected by Render's fromService (e.g. credit-scoring-api-staging.onrender.com)
API_URL = os.getenv("API_URL", "http://localhost:8000")
if not API_URL.startswith(("http://", "https://")):
    API_URL = f"https://{API_URL}"

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .st-form {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .risk-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: white;
        font-weight: bold;
    }
    .low-risk {
        background-color: #4CAF50;
    }
    .medium-risk {
        background-color: #FF9800;
    }
    .high-risk {
        background-color: #F44336;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def call_api(endpoint: str, method: str = "GET", json: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Make API call to FastAPI backend.
    
    Args:
        endpoint: API endpoint (e.g., "/predict", "/health")
        method: HTTP method (GET, POST, etc.)
        json: JSON data for POST requests
        
    Returns:
        Dictionary with response data
        
    Raises:
        Exception: If API call fails
    """
    url = f"{API_URL}{endpoint}"
    
    try:
        # 30s: the API runs on the Render free plan and can be asleep;
        # waking it up takes up to ~1 min, so 10s timed out on cold starts
        if method.upper() == "GET":
            response = requests.get(url, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=json, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        st.error(f"URL: {url}")
        if isinstance(e, requests.exceptions.Timeout):
            st.info(
                "The API may be waking up (Render free plan services sleep "
                "after inactivity and take up to a minute to restart). "
                "Please retry in a minute."
            )
        raise


def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        result = call_api("/health")
        return result.get("model_loaded", False)
    except:
        return False


@st.cache_data(ttl=300)
def fetch_schema(api_url: str) -> Dict[str, Any]:
    """Fetch the raw input schema from the API (/schema endpoint).

    The schema is generated at training time (api/train_model.py) from the
    raw Home Credit columns: field types, form groups, defaults (training
    median/mode) and categorical choices.
    """
    result = call_api("/schema")
    return result if isinstance(result, dict) and "fields" in result else {}


@st.cache_data(ttl=300)
def fetch_threshold(api_url: str) -> float:
    """Fetch the model decision threshold from the API (/model endpoint).

    The threshold is optimized for F1 at training time and stored in the
    model artifact. The API derives its risk categories from it, so the
    dashboard uses the same value for its gauge and recommendations.
    """
    try:
        result = call_api("/model")
        return float(result.get("training_metadata", {}).get("best_threshold") or 0.5)
    except Exception:
        return 0.5


@st.cache_data(ttl=60)
def fetch_monitoring(api_url: str) -> Optional[Dict[str, Any]]:
    """Fetch live serving metrics from the API (/monitoring endpoint)."""
    try:
        result = call_api("/monitoring")
        return result if isinstance(result, dict) and "total_predictions" in result else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_model_info(api_url: str) -> Optional[Dict[str, Any]]:
    """Fetch model metadata from the API (/model endpoint)."""
    try:
        result = call_api("/model")
        return result if isinstance(result, dict) and "model_loaded" in result else None
    except Exception:
        return None


# Display order of the form groups (most relevant first)
GROUP_ORDER = [
    "Client", "Prêt", "Sources externes", "Région",
    "Contact", "Entourage", "Bureau de crédit", "Documents", "Habitat",
]


def render_field(field: Dict[str, Any], override: Any = None):
    """Render a single schema field as a Streamlit widget.

    `override` re-seeds the widget with the user's last submitted value
    (Streamlit resets form widgets to their defaults after each submission).
    """
    if field["type"] == "categorical":
        choices = field.get("choices") or [field.get("default", "")]
        default = override if override is not None else field.get("default")
        index = choices.index(default) if default in choices else 0
        return st.selectbox(field["description"], options=choices, index=index)
    else:
        default = override if override is not None else field.get("default")
        default = float(default) if default is not None else 0.0
        return st.number_input(field["description"], value=default)


def build_dynamic_form(schema: Dict[str, Any],
                        overrides: Optional[Dict[str, Any]] = None,
                        fields_filter: Optional[list] = None) -> Dict[str, Any]:
    """Render the input form as tabs by group. Returns {field_name: value}.

    `fields_filter` optionally restricts the rendered fields;
    omitted fields are imputed with the training median/mode by the API.
    """
    fields = schema["fields"]
    if fields_filter:
        fields = [f for f in fields if f["name"] in fields_filter]
    groups: Dict[str, list] = {}
    for field in fields:
        groups.setdefault(field["group"], []).append(field)

    group_names = [g for g in GROUP_ORDER if g in groups]
    group_names += [g for g in groups if g not in GROUP_ORDER]

    values: Dict[str, Any] = {}
    tabs = st.tabs(group_names)
    for tab, group_name in zip(tabs, group_names):
        with tab:
            cols = st.columns(2)
            for i, field in enumerate(groups[group_name]):
                with cols[i % 2]:
                    values[field["name"]] = render_field(
                        field, override=(overrides or {}).get(field["name"])
                    )
    return values


def display_risk_badge(risk_category: str) -> None:
    """Display a colored risk badge based on the risk category."""
    risk_classes = {
        "LOW_RISK": ("🟢 LOW RISK", "#4CAF50", "low-risk"),
        "MEDIUM_RISK": ("🟡 MEDIUM RISK", "#FF9800", "medium-risk"),
        "HIGH_RISK": ("🔴 HIGH RISK", "#F44336", "high-risk")
    }
    
    if risk_category in risk_classes:
        text, color, css_class = risk_classes[risk_category]
        st.markdown(f"""
        <div class="risk-box {css_class}" style="background-color: {color};">
            <h2 style="margin:0; text-align:center;">{text}</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"Unknown risk category: {risk_category}")


def display_prediction_probability(probability: float, threshold: float = 0.5) -> None:
    """Display a gauge showing the probability of default.

    The gauge zones mirror the API's risk categories: they are relative to
    the model's decision threshold (optimized at training time), not to
    absolute percentages.
    """
    low = threshold * 0.33 * 100
    medium = threshold * 0.67 * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Probability of Default (%) — decision threshold: {threshold * 100:.1f}%"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, low], 'color': "#4CAF50"},
                {'range': [low, medium], 'color': "#FF9800"},
                {'range': [medium, 100], 'color': "#F44336"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=16)
    )
    
    st.plotly_chart(fig, width="stretch")


def render_monitoring_tab() -> None:
    """Live monitoring: real score distribution, latency and risk categories."""
    st.header("📡 Live Monitoring")
    st.markdown(
        "Real serving metrics aggregated by the API (`GET /monitoring`) from "
        "its in-memory buffer of recent predictions — no simulated data."
    )

    monitoring = fetch_monitoring(API_URL)
    if monitoring is None:
        st.error(
            "Could not load monitoring data from the API (`/monitoring` endpoint). "
            "Make sure the FastAPI backend is running."
        )
        return

    if monitoring.get("total_predictions", 0) == 0:
        st.info(
            "No predictions recorded in the API buffer yet. Make predictions "
            "from the **Prediction Form** tab, then refresh this page. "
            "(The buffer is in-memory and resets when the API restarts.)"
        )
        return

    preds = monitoring.get("predictions", [])
    df = pd.DataFrame(preds)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Predictions (buffer)",
            monitoring["total_predictions"],
            f"of {monitoring['buffer_capacity']} max",
        )
    with col2:
        st.metric("Avg inference time", f"{monitoring['avg_latency_ms']:.1f} ms")
    with col3:
        st.metric("P95 inference time", f"{monitoring['p95_latency_ms']:.1f} ms")
    with col4:
        st.metric("Default rate", f"{(monitoring['default_rate'] or 0) * 100:.1f} %")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of predicted scores")
        fig = px.histogram(
            df, x="probability", nbins=20,
            labels={"probability": "Predicted probability of default"},
            color_discrete_sequence=["#4a6fa5"],
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Risk categories (real predictions)")
        risk_df = pd.DataFrame(
            [{"category": k, "count": v} for k, v in monitoring.get("risk_counts", {}).items()]
        )
        fig = px.pie(
            risk_df, values="count", names="category",
            color="category",
            color_discrete_map={
                "LOW_RISK": "#4CAF50",
                "MEDIUM_RISK": "#FF9800",
                "HIGH_RISK": "#F44336",
            },
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Inference latency over time")
    fig = px.scatter(
        df, x="timestamp", y="processing_time_ms",
        labels={"processing_time_ms": "Inference time (ms)"},
    )
    fig.add_hline(
        y=monitoring["avg_latency_ms"], line_dash="dash",
        annotation_text=f"avg {monitoring['avg_latency_ms']:.1f} ms",
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "In-memory buffer: metrics reset on API restart or redeploy. "
        "Persistent tracking of every prediction is available through MLflow."
    )


def render_analytics_tab() -> None:
    """Model analytics: real feature importances and training metrics."""
    st.header("📊 Model Analytics")

    info = fetch_model_info(API_URL)
    if info is None:
        st.error(
            "Could not load model metadata from the API (`/model` endpoint). "
            "Make sure the FastAPI backend is running and the model is loaded."
        )
        return

    meta = info.get("training_metadata") or {}
    best_f1 = meta.get("best_f1")
    threshold = meta.get("best_threshold")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Model type", info.get("model_type") or "N/A")
    with col2:
        st.metric("Best CV F1", f"{best_f1:.3f}" if best_f1 is not None else "N/A")
    with col3:
        st.metric("Decision threshold", f"{threshold:.4f}" if threshold is not None else "N/A")
    with col4:
        st.metric("Engineered features", info.get("feature_count", 0))
    with col5:
        coverage = info.get("form_coverage_pct")
        st.metric(
            "Form fields coverage",
            f"{coverage:.1f}%" if coverage is not None else "N/A",
            help="Share of the model's total gain importance attributable to "
            "the 22 raw columns shown in the simplified prediction form.",
        )

    if best_f1 is not None and threshold is not None:
        st.info(
            "These are the training metrics: F1 is "
            "measured on cross-validated predictions of the default class, which "
            f"represents only ~8% of the dataset — a hard, imbalanced problem. The "
            f"decision threshold ({threshold:.4f}) was optimized for this F1 and "
            "drives the API's risk categories."
        )

    if coverage is not None:
        st.caption(
            f"**Form fields coverage ({coverage:.1f}%)**: the model is trained "
            "on exactly the raw columns shown in the prediction form — the "
            "form, the /schema contract and the model are aligned by "
            "construction. A cross-validation experiment measured that "
            "restricting training to these 22 columns preserves ~99.7% of the "
            "full 120-column model's AUC. This metric is computed by the API "
            "from the deployed model and logged to MLflow at training time."
        )

    st.subheader("Feature importance (real, gain-based)")
    importances = info.get("feature_importances") or []
    if importances:
        fi_df = pd.DataFrame(importances).sort_values("importance_pct", ascending=True)
        fig = px.bar(
            fi_df, x="importance_pct", y="feature", orientation="h",
            labels={
                "importance_pct": "Importance (% of top-20 gain)",
                "feature": "Feature",
            },
            color="importance_pct",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=500, showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Computed live from the deployed model (gain-based importances, "
            f"top 20 of its {info.get('feature_count', 0)} engineered features)."
        )
    else:
        st.info("Feature importances are not available from the loaded model.")

    st.markdown("---")
    st.subheader("Model & run metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset version", meta.get("dataset_version") or "N/A")
        st.metric("Model version", info.get("model_version") or "N/A")
    with col2:
        st.metric("Raw input fields", info.get("raw_input_fields", 0))
        st.metric("MLflow run", (info.get("mlflow") or {}).get("run_id") or "N/A")


def display_client_summary(input_data: Dict[str, Any]) -> None:
    """Display a summary of the client's raw Home Credit application."""
    st.subheader("👤 Client Summary")

    def get(name, default=None):
        return input_data.get(name, default)

    age_years = -get("DAYS_BIRTH", 0) / 365.25
    employed_years = -get("DAYS_EMPLOYED", 0) / 365.25

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Âge", f"{age_years:.0f} ans")
        st.metric("Revenu annuel", f"{get('AMT_INCOME_TOTAL', 0):,.0f}")
        st.metric("Montant du crédit", f"{get('AMT_CREDIT', 0):,.0f}")

    with col2:
        st.metric("Annuité", f"{get('AMT_ANNUITY', 0):,.0f}")
        st.metric("Ancienneté emploi", f"{employed_years:.0f} ans")
        st.metric("Enfants", get("CNT_CHILDREN", 0))

    with col3:
        st.metric("Score externe 1", f"{get('EXT_SOURCE_1', 0):.3f}")
        st.metric("Score externe 2", f"{get('EXT_SOURCE_2', 0):.3f}")
        st.metric("Score externe 3", f"{get('EXT_SOURCE_3', 0):.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Type de revenus", get("NAME_INCOME_TYPE", "N/A"))
        st.metric("Niveau d'études", get("NAME_EDUCATION_TYPE", "N/A"))

    with col2:
        st.metric("Situation familiale", get("NAME_FAMILY_STATUS", "N/A"))
        st.metric("Profession", get("OCCUPATION_TYPE", "N/A"))


def main():
    """Main application function."""
    global API_URL

    # Title
    st.title("💳 Credit Scoring Dashboard")
    st.markdown("""
    This dashboard allows you to evaluate credit risk for potential borrowers using 
    a machine learning model. Enter client information and get an instant risk assessment.
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### API Settings")
    
    # Allow user to override API URL
    custom_api_url = st.sidebar.text_input(
        "API URL",
        value=API_URL,
        help="URL of the FastAPI backend (e.g., http://localhost:8000 or your deployed URL)"
    )
    
    if custom_api_url:
        API_URL = custom_api_url
    
    # Check API health
    st.sidebar.markdown("### API Status")
    if check_api_health():
        st.sidebar.success("✅ API is healthy and model is loaded")
    else:
        st.sidebar.error("❌ API is not responding or model is not loaded")
        st.sidebar.warning("Please start the FastAPI server first:")
        st.sidebar.code("uvicorn api.main:app --reload", language="bash")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **Credit Scoring Model v1.0.0**

    This model predicts the probability of loan default from the raw
    Home Credit application fields:
    - Client demographics & employment
    - Loan amounts and annuity
    - External source scores (EXT_SOURCE_1-3)
    - Region, housing, credit bureau data

    Risk categories (relative to the model's decision threshold,
    optimized at training time — see /model):
    - 🟢 LOW: probability < threshold × 0.33
    - 🟡 MEDIUM: below threshold × 0.67
    - 🔴 HIGH: at or above threshold × 0.67
    """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Prediction Form", "📡 Monitoring", "📊 Model Analytics", "ℹ️ Help"])
    
    with tab1:
        # Prediction Form (generated dynamically from the API /schema endpoint)
        st.header("Client Information Form")
        st.markdown("""
        This form is generated from the raw input schema served by the API
        (`/schema`): the model is trained on exactly these 22 columns, so what
        you see is what the model uses. Each field defaults to the training
        median/mode, so you only need to adjust the fields you care about.
        """)

        schema = fetch_schema(API_URL)
        if not schema:
            st.error(
                "Could not load the input schema from the API (`/schema` endpoint). "
                "Make sure the FastAPI backend is running and the model is loaded."
            )
        else:
            # Re-seed the form with the last submitted values: st.form resets
            # widgets to their defaults on every submission, which made
            # consecutive predictions look "static" (same defaults resent).
            prior_input = st.session_state.get("last_submitted_input", {})
            with st.form("prediction_form"):
                # The schema served by the API IS the model contract: the
                # model is trained on exactly the columns in /schema.
                input_data = build_dynamic_form(schema, overrides=prior_input)

                # Submit button
                st.markdown("---")
                submitted = st.form_submit_button(
                    "🔍 Get Credit Score",
                    width="stretch"
                )

        # Process prediction
        if schema and submitted:
            # Remember the submitted values for the next form render
            st.session_state["last_submitted_input"] = input_data
            # Display loading spinner
            with st.spinner("Analyzing credit risk..."):
                try:
                    # Call API
                    result = call_api("/predict", method="POST", json=input_data)

                    # Display results
                    st.success("✅ Prediction Successful!")

                    # Show client summary
                    display_client_summary(input_data)

                    st.markdown("---")
                    st.header("📈 Prediction Results")

                    # Display risk category
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        display_risk_badge(result.get("risk_category", "UNKNOWN"))

                    # Display probability gauge (zones relative to the
                    # model's decision threshold, like the API categories)
                    decision_threshold = fetch_threshold(API_URL)
                    display_prediction_probability(
                        result.get("probability", 0.5),
                        threshold=decision_threshold,
                    )

                    # Display detailed results
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Prediction",
                            "Default" if result.get("prediction") == 1 else "No Default"
                        )

                    with col2:
                        st.metric(
                            "Probability",
                            f"{result.get('probability', 0) * 100:.2f}%"
                        )

                    with col3:
                        st.metric("Decision threshold", f"{decision_threshold * 100:.2f}%")

                    with col4:
                        st.metric("Model Version", result.get("model_version", "N/A"))

                    st.caption(
                        f"The prediction compares the probability to the decision "
                        f"threshold (optimized for F1 at training time): below "
                        f"{decision_threshold:.2%} -> No Default, above -> Default. "
                        f"Risk bands: LOW < {decision_threshold * 0.33:.1%}, "
                        f"MEDIUM < {decision_threshold * 0.67:.1%}, "
                        f"HIGH >= {decision_threshold * 0.67:.1%}."
                    )

                    # Recommendation (driven by the API's risk_category,
                    # which is relative to the training threshold)
                    st.markdown("---")
                    st.subheader("💡 Recommendation")

                    risk_category = result.get("risk_category", "HIGH_RISK")

                    if risk_category == "LOW_RISK":
                        st.success("""
                        **APPROVE** ✅

                        This client has a low risk of default. The loan can be approved
                        with standard terms and conditions.
                        """)
                    elif risk_category == "MEDIUM_RISK":
                        st.info("""
                        **APPROVE WITH CAUTION** ⚠️

                        This client has a moderate risk of default. Consider:
                        - Higher interest rate
                        - Additional collateral
                        - Smaller loan amount
                        """)
                    else:
                        st.error("""
                        **REJECT** ❌

                        This client has a high risk of default. The loan should be rejected
                        or require significant modifications:
                        - Very high interest rate
                        - Full collateral coverage
                        - Co-signer required
                        """)

                except Exception as e:
                    st.error(f"Failed to get prediction: {str(e)}")

    with tab2:
        render_monitoring_tab()

    with tab3:
        render_analytics_tab()

    with tab4:
        # Help Tab
        st.header("ℹ️ Help & Documentation")
        
        st.markdown("""
        ## How to Use This Dashboard
        
        ### 1. Enter Client Information
        Fill out the form in the **Prediction Form** tab with the client's
        Home Credit application fields, grouped in tabs:
        - Client (demographics, employment, income)
        - Loan (amount, annuity, goods price)
        - External sources (EXT_SOURCE scores)
        - Region, contact, credit bureau

        Each field is pre-filled with the training median/mode, so a prediction
        can be requested without filling everything.

        ### 2. Get Prediction
        Click the "Get Credit Score" button to send the data to the API and receive:
        - Risk category (LOW, MEDIUM, HIGH)
        - Probability of default
        - Recommendation

        ### 3. Interpret Results

        **Risk Categories:**
        Categories are computed by the API relative to the model's decision
        threshold (optimized for F1 at training time, visible on `/model`):
        - 🟢 **LOW RISK**: probability < threshold × 0.33 — safe to approve
        - 🟡 **MEDIUM RISK**: below threshold × 0.67 — approve with caution
        - 🔴 **HIGH RISK**: at or above threshold × 0.67 — likely to default

        **Recommendations:**
        - Based on the risk category, the dashboard provides actionable advice

        ## API Endpoints

        The FastAPI backend provides the following endpoints:

        | Method | Endpoint | Description |
        |--------|----------|-------------|
        | GET | `/` | API information |
        | GET | `/health` | Health check |
        | GET | `/model` | Model metadata |
        | GET | `/schema` | Raw input schema (drives this form) |
        | POST | `/predict` | Make prediction |
        | GET | `/monitoring` | Live serving metrics (scores, latency, risk categories) |
        | GET | `/docs` | Swagger documentation |
        | GET | `/redoc` | ReDoc documentation |
        | GET | `/openapi.json` | OpenAPI schema |

        ## Example API Request

        ```python
        import requests

        url = "http://localhost:8000/predict"
        # Raw Home Credit application fields; omitted fields are filled
        # with the training median/mode by the API
        data = {
            "AMT_INCOME_TOTAL": 202500.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 30000.0,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -1500,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.7,
            "CODE_GENDER": "M",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
        }

        response = requests.post(url, json=data)
        print(response.json())
        ```

        ## Technical Details

        **Model:** LightGBM (or LogisticRegression / RandomForest) trained on
        the Home Credit `application_train` dataset via the pipeline in
        `api/ml_pipeline.py` (polynomial + domain features, label and
        one-hot encoding). The full list of its 152 engineered features is
        available on `/model`.

        **Input:** the 22 raw application columns shown in the prediction form
        (all optional, see `/schema`).
        The preprocessing fitted at training time is applied by the API before
        prediction, which guarantees train/serve consistency.

        **Output:**
        - Prediction (0 or 1)
        - Probability (0.0 to 1.0)
        - Risk Category (LOW_RISK, MEDIUM_RISK, HIGH_RISK)
        - Model Version

        ## Troubleshooting

        **API not responding:**
        - Make sure the FastAPI server is running
        - Check that the API URL in the sidebar is correct
        - Verify that the model file exists at `api/models/credit_scoring_model.pkl`

        **Form not loading:**
        - The schema is fetched from the API `/schema` endpoint: the API must
          be reachable for the form to be generated
        - Retrying after a backend restart requires a cache reset (the schema
          is cached for 5 minutes)

        **Invalid input errors:**
        - Numeric fields must contain numbers
        - Numeric fields are range-checked by the API (e.g. external scores
          must lie in [0, 1], age in 18-110 years, amounts non-negative):
          out-of-range values are rejected with a 422 error
        - Categorical fields accept any string, but values outside the
          training choices have no learned encoding (mapped to a neutral value)

        **Docker issues:**
        - Make sure Docker is installed and running
        - Check that ports are not blocked by firewalls
        - Verify that the Docker image was built successfully

        ## Contact

        For questions or issues, please refer to the project documentation or
        contact the MLOps team.
        """)


if __name__ == "__main__":
    main()
