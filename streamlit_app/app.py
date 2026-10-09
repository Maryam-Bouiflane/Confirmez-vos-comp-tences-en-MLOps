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
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=json, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        st.error(f"URL: {url}")
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


# Display order of the form groups (most relevant first)
GROUP_ORDER = [
    "Client", "Prêt", "Sources externes", "Région",
    "Contact", "Entourage", "Bureau de crédit", "Documents", "Habitat",
]


def render_field(field: Dict[str, Any]):
    """Render a single schema field as a Streamlit widget."""
    if field["type"] == "categorical":
        choices = field.get("choices") or [field.get("default", "")]
        default = field.get("default")
        index = choices.index(default) if default in choices else 0
        return st.selectbox(field["description"], options=choices, index=index)
    else:
        default = field.get("default")
        default = float(default) if default is not None else 0.0
        return st.number_input(field["description"], value=default)


def build_dynamic_form(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Render the input form as tabs by group. Returns {field_name: value}."""
    groups: Dict[str, list] = {}
    for field in schema["fields"]:
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
                    values[field["name"]] = render_field(field)
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
        title={'text': "Probability of Default (%)"},
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
    
    st.plotly_chart(fig, use_container_width=True)


def display_feature_importance() -> None:
    """Display feature importance visualization (placeholder)."""
    st.subheader("📊 Feature Importance")
    
    # This is a placeholder - in a real app, you would load this from the model
    features = [
        "Credit Score", "Debt to Income", "Previous Default", "Loan Amount",
        "Credit History", "Annual Income", "Employment Years", "Age"
    ]
    importance = [0.25, 0.20, 0.18, 0.12, 0.10, 0.08, 0.05, 0.02]
    
    df_importance = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values("Importance", ascending=True)
    
    fig = px.bar(
        df_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance for Credit Scoring Model",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig, use_container_width=True)


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
    tab1, tab2, tab3 = st.tabs(["📝 Prediction Form", "📊 Analytics", "ℹ️ Help"])
    
    with tab1:
        # Prediction Form (generated dynamically from the API /schema endpoint)
        st.header("Client Information Form")
        st.markdown("""
        This form is generated from the raw Home Credit schema served by the API
        (`/schema`). Each field defaults to the training median/mode, so you only
        need to adjust the fields you care about before requesting a prediction.
        """)

        schema = fetch_schema(API_URL)
        if not schema:
            st.error(
                "Could not load the input schema from the API (`/schema` endpoint). "
                "Make sure the FastAPI backend is running and the model is loaded."
            )
        else:
            with st.form("prediction_form"):
                input_data = build_dynamic_form(schema)

                # Submit button
                st.markdown("---")
                submitted = st.form_submit_button(
                    "🔍 Get Credit Score",
                    use_container_width=True
                )

        # Process prediction
        if schema and submitted:
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
                    display_prediction_probability(
                        result.get("probability", 0.5),
                        threshold=fetch_threshold(API_URL),
                    )

                    # Display detailed results
                    col1, col2, col3 = st.columns(3)

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
                        st.metric("Model Version", result.get("model_version", "N/A"))

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
        # Analytics Tab
        st.header("📊 Model Analytics")
        
        st.markdown("""
        Explore the credit scoring model's behavior and feature importance.
        """)
        
        # Display feature importance
        display_feature_importance()
        
        st.markdown("---")
        
        # Risk Distribution (simulated)
        st.subheader("📊 Simulated Risk Distribution")
        
        np.random.seed(42)
        risk_categories = ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]
        counts = [500, 300, 200]
        
        fig = px.pie(
            values=counts,
            names=risk_categories,
            title="Distribution of Risk Categories (Simulated Data)",
            color=risk_categories,
            color_discrete_map={
                "LOW_RISK": "#4CAF50",
                "MEDIUM_RISK": "#FF9800",
                "HIGH_RISK": "#F44336"
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Model performance metrics (simulated)
        st.subheader("🎯 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "92.5%")
        with col2:
            st.metric("Precision", "88.0%")
        with col3:
            st.metric("Recall", "90.2%")
        with col4:
            st.metric("F1-Score", "89.1%")
    
    with tab3:
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
        - Region, contact, entourage, credit bureau, documents, housing

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
        one-hot encoding). The full list of ~280 engineered features is
        available on `/model`.

        **Input:** the 120 raw application columns (all optional, see `/schema`).
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
