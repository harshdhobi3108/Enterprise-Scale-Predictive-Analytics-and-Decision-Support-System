"""
Customer Lifecycle Intelligence Dashboard
Retention Prediction + SHAP Explainability
(Fully Fixed Version - Clean Customer Mapping)
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
# Load Retention Features + Customer Info
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

    # Build features
    features = build_retention_features(orders, payments, reviews)

    # ==========================================================
    # FIXED CUSTOMER MASTER TABLE (NO DUPLICATES)
    # ==========================================================
    customer_info = (
        customers
        .sort_values("customer_unique_id")
        .drop_duplicates("customer_unique_id")
        .reset_index(drop=True)
    )

    # Create clean business ID
    customer_info["customer_code"] = (
        "CUST-" + (customer_info.index + 1).astype(str).str.zfill(5)
    )

    return features, customer_info


# ==========================================================
# Dashboard
# ==========================================================
def run_retention_dashboard():

    st.title("Customer Lifecycle Intelligence")

    # ---------------------------------------------------------
    # Load Model (compatible with old + new)
    # ---------------------------------------------------------
    model_bundle = joblib.load(MODEL_PATH)

    if isinstance(model_bundle, dict):
        model = model_bundle["model"]
        threshold = model_bundle.get("threshold", 0.5)
    else:
        model = model_bundle
        threshold = 0.5

    explainer = RetentionExplainer(MODEL_PATH)

    # ---------------------------------------------------------
    # Load Data
    # ---------------------------------------------------------
    features_df, customer_info = load_retention_features()

    # ---------------------------------------------------------
    # Merge Customer Info (CLEAN JOIN)
    # ---------------------------------------------------------
    features_df = features_df.merge(
        customer_info[
            ["customer_unique_id", "customer_code", "customer_city", "customer_state"]
        ],
        on="customer_unique_id",
        how="left"
    )

    # ---------------------------------------------------------
    # Create Display Column (MATCHES FIRST PAGE)
    # ---------------------------------------------------------
    features_df["customer_display"] = (
        features_df["customer_code"] +
        " | " +
        features_df["customer_city"].str.title() +
        " | " +
        features_df["customer_state"]
    )

    # ---------------------------------------------------------
    # Compute Probabilities
    # ---------------------------------------------------------
    X_all = features_df.drop(columns=[
        "customer_unique_id",
        "customer_code",
        "customer_city",
        "customer_state",
        "customer_display"
    ])

    probabilities = model.predict_proba(X_all)[:, 1]
    features_df["retention_probability"] = probabilities

    # ---------------------------------------------------------
    # Top Customers
    # ---------------------------------------------------------
    st.subheader("Top Likely To Retain Customers")

    top_customers = (
        features_df
        .sort_values("retention_probability", ascending=False)
        .drop_duplicates(subset=["customer_code"])
        .head(5)
    )

    st.dataframe(
        top_customers[
            ["customer_display", "retention_probability"]
        ].rename(columns={
            "customer_display": "Customer",
            "retention_probability": "Retention Probability"
        })
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # Customer Selection
    # ---------------------------------------------------------
    customer_map = dict(
        zip(features_df["customer_display"], features_df["customer_unique_id"])
    )

    selected_display = st.selectbox(
        "Select Customer",
        list(customer_map.keys())
    )

    selected_customer = customer_map[selected_display]

    # ---------------------------------------------------------
    # Prepare Customer Data
    # ---------------------------------------------------------
    customer_data = features_df[
        features_df["customer_unique_id"] == selected_customer
    ].drop(columns=[
        "customer_unique_id",
        "customer_code",
        "customer_city",
        "customer_state",
        "customer_display",
        "retention_probability"
    ])

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    probability = model.predict_proba(customer_data)[0][1]

    st.metric("Retention Probability (%)", f"{probability * 100:.2f}")

    if probability < threshold:
        st.error("High Churn Risk")
    else:
        st.success("Likely to Retain")

    # ---------------------------------------------------------
    # LOCAL SHAP
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

    sample_data = X_all.sample(min(1000, len(X_all)))
    shap_values_global, X_global = explainer.explain_global(sample_data)

    fig_global = plt.figure()
    shap.summary_plot(
        shap_values_global,
        X_global,
        feature_names=X_global.columns,
        show=False
    )
    st.pyplot(fig_global)