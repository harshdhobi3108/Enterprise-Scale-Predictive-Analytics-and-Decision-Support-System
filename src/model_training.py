"""
Enterprise Delivery Delay Prediction System
Production-Grade Professional Dashboard
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Enterprise Delivery Delay System",
    layout="wide"
)


# --------------------------------------------------
# Resource Loading (Cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/delivery_delay_model.pkl")


@st.cache_data
def load_feature_importance():
    return pd.read_csv("models/feature_importance.csv")


@st.cache_resource
def load_explainer(model):
    return shap.TreeExplainer(model)


model = load_model()
importance_df = load_feature_importance()
explainer = load_explainer(model)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Enterprise Delivery Delay Prediction System")
st.markdown("Predictive Analytics and Decision Support Dashboard")

st.divider()


# --------------------------------------------------
# Sidebar – Input Parameters
# --------------------------------------------------
st.sidebar.header("Order Parameters")

purchase_hour = st.sidebar.slider("Purchase Hour", 0, 23, 12)
purchase_dayofweek = st.sidebar.slider("Purchase Day of Week (0 = Monday)", 0, 6, 2)
purchase_month = st.sidebar.slider("Purchase Month", 1, 12, 6)

approval_delay_hours = st.sidebar.number_input(
    "Approval Delay (hours)", 0.0, 200.0, 2.0
)

carrier_delay_hours = st.sidebar.number_input(
    "Carrier Delay (hours)", 0.0, 500.0, 12.0
)

estimated_delivery_days = st.sidebar.number_input(
    "Estimated Delivery Days", 1.0, 60.0, 7.0
)

total_payment_value = st.sidebar.number_input(
    "Total Payment Value", 0.0, 10000.0, 150.0
)

payment_installments = st.sidebar.slider(
    "Payment Installments", 1, 24, 1
)


# --------------------------------------------------
# Input Data Preparation
# --------------------------------------------------
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


# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.subheader("Delay Risk Prediction")

probability = float(model.predict_proba(input_data)[0][1])

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Delay Probability", f"{probability:.2%}")

with col2:
    if probability >= 0.60:
        st.error("High Risk")
    elif probability >= 0.30:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")


# --------------------------------------------------
# Risk Gauge
# --------------------------------------------------
st.subheader("Risk Confidence Indicator")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100,
    title={'text': "Delay Risk (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkred"},
        'steps': [
            {'range': [0, 30], 'color': "#d4edda"},
            {'range': [30, 60], 'color': "#fff3cd"},
            {'range': [60, 100], 'color': "#f8d7da"},
        ],
    }
))

st.plotly_chart(gauge, use_container_width=True)

st.divider()


# --------------------------------------------------
# SHAP Explainability – Robust Implementation
# --------------------------------------------------
st.subheader("Explainable AI – Feature Contributions")

try:
    shap_values = explainer.shap_values(input_data)

    # Robust handling of SHAP output format
    if isinstance(shap_values, list):
        # If multiple classes exist, use positive class if available
        if len(shap_values) > 1:
            shap_array = shap_values[1][0]
        else:
            shap_array = shap_values[0][0]
    else:
        shap_array = shap_values[0]

    shap_df = (
        pd.DataFrame({
            "Feature": input_data.columns,
            "SHAP Value": shap_array
        })
        .assign(Absolute_Impact=lambda df: df["SHAP Value"].abs())
        .sort_values("Absolute_Impact", ascending=False)
        .drop(columns="Absolute_Impact")
        .reset_index(drop=True)
    )

    st.dataframe(shap_df, use_container_width=True)

except Exception as error:
    st.error("Explainability module could not be generated.")
    st.text(str(error))


st.divider()


# --------------------------------------------------
# Global Feature Importance
# --------------------------------------------------
st.subheader("Global Model Feature Importance")

importance_chart = px.bar(
    importance_df,
    x="importance",
    y="feature",
    orientation="h",
    title="Feature Importance Ranking"
)

st.plotly_chart(importance_chart, use_container_width=True)

st.divider()


# --------------------------------------------------
# Model Overview
# --------------------------------------------------
st.subheader("Model Overview")

st.markdown("""
Model Type: LightGBM Gradient Boosting  
Validation Strategy: Time-aware Cross Validation  
Class Imbalance Handling: scale_pos_weight applied  
Threshold Optimization: F1-optimized  
Current AUC: ~0.77  

This system supports proactive logistics risk management and operational decision-making.
""")
