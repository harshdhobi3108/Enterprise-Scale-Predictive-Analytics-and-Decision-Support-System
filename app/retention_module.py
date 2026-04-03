"""
Customer Lifecycle Intelligence Dashboard
Retention Prediction + SHAP Explainability
(FINAL FIXED VERSION - STREAMLIT SAFE)
"""

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# SAFE SHAP IMPORT
# ==========================================================
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

from src.data_loader import DataLoader
from src.retention_features import build_retention_features
from src.retention_explainer import RetentionExplainer

MODEL_PATH = "models/retention_model.pkl"


# ==========================================================
# LOAD DATA
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

    # Clean customer table
    customer_info = (
        customers
        .sort_values("customer_unique_id")
        .drop_duplicates("customer_unique_id")
        .reset_index(drop=True)
    )

    # Business-friendly ID
    customer_info["customer_code"] = (
        "CUST-" + (customer_info.index + 1).astype(str).str.zfill(5)
    )

    return features, customer_info


# ==========================================================
# MAIN DASHBOARD
# ==========================================================
def run_retention_dashboard():

    st.title("Customer Lifecycle Intelligence")

    # ---------------------------------------------------------
    # LOAD MODEL
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
    # LOAD DATA
    # ---------------------------------------------------------
    features_df, customer_info = load_retention_features()

    features_df = features_df.merge(
        customer_info[
            ["customer_unique_id", "customer_code", "customer_city", "customer_state"]
        ],
        on="customer_unique_id",
        how="left"
    )

    features_df["customer_display"] = (
        features_df["customer_code"] +
        " | " +
        features_df["customer_city"].str.title() +
        " | " +
        features_df["customer_state"]
    )

    # ---------------------------------------------------------
    # PREPARE FEATURES
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
    # TOP CUSTOMERS
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
        }),
        use_container_width=True
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # CUSTOMER SELECTION
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
    # CUSTOMER DATA
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
    # PREDICTION
    # ---------------------------------------------------------
    probability = model.predict_proba(customer_data)[0][1]

    st.metric("Retention Probability (%)", f"{probability * 100:.2f}")

    if probability < threshold:
        st.error("High Churn Risk")
    else:
        st.success("Likely to Retain")

    # ==========================================================
    # SHAP - LOCAL EXPLANATION (FIXED)
    # ==========================================================
    st.subheader("Why this prediction?")

    if SHAP_AVAILABLE:
        try:
            shap_values, X_named = explainer.explain_instance(customer_data)

            if shap_values is not None:

                expected_value = explainer.explainer.expected_value[1]
                shap_val = shap_values[1][0]
                features = X_named.iloc[0]

                fig, ax = plt.subplots()

                shap.plots._waterfall.waterfall_legacy(
                    expected_value,
                    shap_val,
                    features,
                    show=False
                )

                st.pyplot(fig)

            else:
                st.warning("SHAP values not available")

        except Exception as e:
            st.warning(f"SHAP error: {e}")

    else:
        st.info("SHAP not installed")

    # ==========================================================
    # SHAP - GLOBAL EXPLANATION (FIXED)
    # ==========================================================
    st.subheader("Global Retention Drivers")

    if SHAP_AVAILABLE:
        try:
            sample_data = X_all.sample(min(500, len(X_all)))

            shap_values_global, X_global = explainer.explain_global(sample_data)

            if shap_values_global is not None:

                fig = plt.figure()

                shap.summary_plot(
                    shap_values_global[1],  # ✅ FIXED
                    X_global,
                    feature_names=X_global.columns,
                    show=False
                )

                st.pyplot(fig)

            else:
                st.warning("Global SHAP not available")

        except Exception as e:
            st.warning(f"Global SHAP error: {e}")

    else:
        st.info("Global SHAP not available")