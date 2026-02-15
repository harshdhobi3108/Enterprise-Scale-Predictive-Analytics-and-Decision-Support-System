"""
Enterprise Delivery Delay Intelligence Platform
Production-Grade Monitoring + Explainability Version
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import os
import threading
from datetime import datetime, timezone


# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Delivery Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Executive Dark Styling
# ==========================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a, #111827); color: white; }
    h1 { font-size: 34px !important; font-weight: 700 !important; color: #f8fafc !important; }
    h2, h3 { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] { background-color: #0b1220; }
    section[data-testid="stSidebar"] label { color: #cbd5e1 !important; }
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 22px;
        border-radius: 14px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# Resource Loading (Safe)
# ==========================================================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("models/delivery_delay_model.pkl")
        explainer = shap.TreeExplainer(model)
        importance_df = pd.read_csv("models/feature_importance.csv")
        return model, explainer, importance_df
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model, explainer, importance_df = load_resources()

# Enforce feature order
EXPECTED_FEATURES = list(model.feature_name_)

# ==========================================================
# Thread Lock for Logging
# ==========================================================
log_lock = threading.Lock()

def log_prediction(input_data, probability, risk_level):
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/prediction_log.csv"

    log_entry = input_data.copy()
    log_entry["probability"] = probability
    log_entry["risk_level"] = risk_level
    log_entry["timestamp"] = datetime.now(timezone.utc)

    with log_lock:
        if os.path.exists(log_file):
            log_entry.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_entry.to_csv(log_file, mode="w", header=True, index=False)


# ==========================================================
# Header
# ==========================================================
st.markdown("""
<div style="padding:15px 0;">
    <h1>Delivery Risk Intelligence Platform</h1>
    <p style="color:#94a3b8;font-size:16px;">
        Enterprise Predictive Analytics | Operational Risk Monitoring | Explainable AI
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.markdown("## Configuration Panel")
st.sidebar.markdown("---")

medium_threshold = st.sidebar.slider(
    "Medium Risk Threshold",
    0.1, 0.6, 0.3, 0.05
)

high_threshold = st.sidebar.slider(
    "High Risk Threshold",
    0.4, 0.95, 0.6, 0.05
)

with st.sidebar.expander("Time Features", expanded=True):
    purchase_hour = st.slider("Purchase Hour", 0, 23, 12)
    purchase_dayofweek = st.slider("Day of Week", 0, 6, 2)
    purchase_month = st.slider("Purchase Month", 1, 12, 6)

with st.sidebar.expander("Logistics Features", expanded=True):
    approval_delay_hours = st.number_input("Approval Delay (hours)", 0.0, 200.0, 2.0)
    carrier_delay_hours = st.number_input("Carrier Delay (hours)", 0.0, 500.0, 12.0)
    estimated_delivery_days = st.number_input("Estimated Delivery Days", 1.0, 60.0, 7.0)

with st.sidebar.expander("Financial Features", expanded=True):
    total_payment_value = st.number_input("Total Payment Value", 0.0, 10000.0, 150.0)
    payment_installments = st.slider("Payment Installments", 1, 24, 1)


# ==========================================================
# Input Data
# ==========================================================
input_data = pd.DataFrame([{
    "purchase_hour": purchase_hour,
    "purchase_dayofweek": purchase_dayofweek,
    "purchase_month": purchase_month,
    "approval_delay_hours": approval_delay_hours,
    "carrier_delay_hours": carrier_delay_hours,
    "estimated_delivery_days": estimated_delivery_days,
    "total_payment_value": total_payment_value,
    "payment_installments": payment_installments,
}])

# Schema validation
if set(input_data.columns) != set(EXPECTED_FEATURES):
    st.error("Feature mismatch between dashboard and trained model.")
    st.stop()

# Reorder safely
input_data = input_data.reindex(columns=EXPECTED_FEATURES)


# ==========================================================
# Prediction Section
# ==========================================================
predict_button = st.button("Generate Risk Assessment")

if predict_button:

    try:
        probability = float(model.predict_proba(input_data)[0][1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Risk logic
    if probability >= high_threshold:
        level = "High"
    elif probability >= medium_threshold:
        level = "Medium"
    else:
        level = "Low"

    st.subheader("Risk Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Delay Probability", f"{probability:.2%}")

    with col2:
        st.metric("Risk Score", f"{probability*100:.1f}/100")

    with col3:
        st.metric("Risk Level", level)

    log_prediction(input_data, probability, level)

    if level == "High":
        st.error("Critical Alert: High likelihood of delivery delay.")
    elif level == "Medium":
        st.warning("Moderate Risk: Monitor logistics performance.")
    else:
        st.success("Operational Status: Risk within acceptable limits.")

    st.divider()

    # ======================================================
    # SHAP Explainability
    # ======================================================
    st.subheader("Explainable AI – Feature Contribution")

    try:
        shap_values = explainer.shap_values(input_data)

        if isinstance(shap_values, list):
            shap_array = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_array = shap_values[0]

        shap_df = (
            pd.DataFrame({
                "feature": EXPECTED_FEATURES,
                "shap_value": shap_array
            })
            .assign(abs_val=lambda x: x["shap_value"].abs())
            .sort_values("abs_val", ascending=False)
            .drop(columns="abs_val")
        )

        shap_chart = px.bar(
            shap_df,
            x="shap_value",
            y="feature",
            orientation="h",
            title="Feature Impact on Current Prediction"
        )

        shap_chart.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font=dict(color="white")
        )

        st.plotly_chart(shap_chart, use_container_width=True)

    except Exception as e:
        st.warning(f"Explainability unavailable: {e}")

st.divider()


# ==========================================================
# Monitoring Dashboard
# ==========================================================
st.subheader("Risk Monitoring Overview")

log_file = "logs/prediction_log.csv"

if os.path.exists(log_file):

    log_df = pd.read_csv(log_file)

    if not log_df.empty and "timestamp" in log_df.columns:

        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], errors="coerce")
        log_df = log_df.dropna(subset=["timestamp"])

        total_predictions = len(log_df)
        avg_risk = log_df["probability"].mean()
        high_risk_count = (log_df["risk_level"] == "High").sum()

        k1, k2, k3 = st.columns(3)

        k1.metric("Total Predictions", total_predictions)
        k2.metric("Average Risk", f"{avg_risk:.2%}")
        k3.metric("High Risk Cases", high_risk_count)

        if avg_risk > high_threshold:
            st.error("System Alert: Average risk critically high.")
        elif high_risk_count >= 3:
            st.warning("Multiple high-risk cases detected.")
        else:
            st.success("Monitoring stable.")

        log_df["minute"] = log_df["timestamp"].dt.floor("min")

        minute_risk = (
            log_df.groupby("minute")["probability"]
            .mean()
            .reset_index()
        )

        trend_chart = px.line(
            minute_risk,
            x="minute",
            y="probability",
            title="Average Risk Trend (Per Minute)"
        )

        trend_chart.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font=dict(color="white")
        )

        st.plotly_chart(trend_chart, use_container_width=True)

    else:
        st.info("Generate predictions to activate monitoring.")
else:
    st.info("No prediction data available yet.")


st.divider()

# ==========================================================
# Global Feature Importance
# ==========================================================
st.subheader("Model Feature Importance")

importance_chart = px.bar(
    importance_df,
    x="importance",
    y="feature",
    orientation="h"
)

importance_chart.update_layout(
    plot_bgcolor="#111827",
    paper_bgcolor="#111827",
    font=dict(color="white")
)

st.plotly_chart(importance_chart, use_container_width=True)

st.divider()

# ==========================================================
# Operational Insights
# ==========================================================
st.subheader("Operational Intelligence Summary")

st.markdown("""
Carrier handling delays significantly elevate risk exposure.  
Extended estimated delivery windows introduce fulfillment uncertainty.  
High-value transactions may require priority logistics intervention.

This platform supports proactive operational decision-making.
""")
