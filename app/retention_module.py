"""
Customer Lifecycle Intelligence Dashboard
Retention Prediction + SHAP Explainability
"""

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.data_loader import DataLoader
from src.retention_features import build_retention_features
from src.retention_explainer import RetentionExplainer


MODEL_PATH = "models/retention_model.pkl"


# ==========================================================
# Load Retention Features
# ==========================================================
@st.cache_data
def load_retention_features():

    loader = DataLoader("data/raw")
    data = loader.load_all()

    orders = data["orders"]
    customers = data["customers"]
    payments = data["payments"]
    reviews = data["reviews"]

    # Attach customer_unique_id
    orders = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left"
    )

    payments = payments.merge(
        orders[["order_id", "customer_unique_id"]],
        on="order_id",
        how="left"
    )

    reviews = reviews.merge(
        orders[["order_id", "customer_unique_id"]],
        on="order_id",
        how="left"
    )

    features = build_retention_features(orders, payments, reviews)

    return features


# ==========================================================
# Dashboard
# ==========================================================
def run_retention_dashboard():

    st.title("Customer Lifecycle Intelligence")

    # ---------------------------------------------------------
    # Load Model & Features
    # ---------------------------------------------------------
    model = joblib.load(MODEL_PATH)
    explainer = RetentionExplainer(MODEL_PATH)

    features_df = load_retention_features()

    # ---------------------------------------------------------
    # Compute Probabilities for All Customers
    # ---------------------------------------------------------
    X_all = features_df.drop(columns=["customer_unique_id"])
    probabilities = model.predict_proba(X_all)[:, 1]

    features_df["retention_probability"] = probabilities

    # ---------------------------------------------------------
    # Show Top Retention Customers
    # ---------------------------------------------------------
    st.subheader("Top Likely To Retain Customers")

    top_customers = (
        features_df
        .sort_values("retention_probability", ascending=False)
        .head(5)
    )

    st.dataframe(
        top_customers[
            ["customer_unique_id", "retention_probability"]
        ]
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # Select Customer
    # ---------------------------------------------------------
    customer_ids = features_df["customer_unique_id"].tolist()

    selected_customer = st.selectbox(
        "Select Customer",
        customer_ids
    )

    customer_data = features_df[
        features_df["customer_unique_id"] == selected_customer
    ].drop(columns=["customer_unique_id", "retention_probability"])

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    probability = model.predict_proba(customer_data)[0][1]

    st.metric("Retention Probability (%)", f"{probability * 100:.2f}")

    if probability < 0.3:
        st.error("High Churn Risk")
    elif probability < 0.6:
        st.warning("Moderate Risk")
    else:
        st.success("Likely to Retain")

    # ---------------------------------------------------------
    # LOCAL SHAP (Waterfall Plot – works better in Streamlit)
    # ---------------------------------------------------------
    st.subheader("Why this prediction?")

    shap_values, X_named = explainer.explain_instance(customer_data)

    fig_local = plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.explainer.expected_value,
        shap_values[0],
        feature_names=X_named.columns
    )
    st.pyplot(fig_local)

    # ---------------------------------------------------------
    # GLOBAL SHAP
    # ---------------------------------------------------------
    st.subheader("Global Retention Drivers")

    sample_data = X_all.sample(1000)
    shap_values_global, X_global = explainer.explain_global(sample_data)

    fig_global = plt.figure()
    shap.summary_plot(
        shap_values_global,
        X_global,
        feature_names=X_global.columns,
        show=False
    )
    st.pyplot(fig_global)
