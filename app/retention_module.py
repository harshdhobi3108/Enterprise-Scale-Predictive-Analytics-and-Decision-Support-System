"""
Retention Dashboard - ELITE VERSION
Enterprise Grade | Business Insights | Clean ML Integration
"""

import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_retention_features():

    from src.data_loader import DataLoader
    from src.retention_features import build_retention_features

    loader = DataLoader("data/raw")
    data = loader.load_all()

    orders = data["orders"]
    customers = data["customers"]
    payments = data["payments"]
    reviews = data["reviews"]

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

    customer_info = (
        customers
        .sort_values("customer_unique_id")
        .drop_duplicates("customer_unique_id")
        .reset_index(drop=True)
    )

    customer_info["customer_code"] = (
        "CUST-" + (customer_info.index + 1).astype(str).str.zfill(5)
    )

    return features, customer_info


# ==========================================================
# MAIN DASHBOARD
# ==========================================================
def run_retention_dashboard():

    st.title("Customer Lifecycle Intelligence")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_PATH = os.path.join(BASE_DIR, "models", "delivery_model.pkl")
    FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")
    THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.pkl")

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    threshold = joblib.load(THRESHOLD_PATH) if os.path.exists(THRESHOLD_PATH) else 0.5

    # ==========================================================
    # LOAD DATA
    # ==========================================================
    features_df, customer_info = load_retention_features()

    features_df = features_df.merge(
        customer_info[
            ["customer_unique_id", "customer_code", "customer_city", "customer_state"]
        ],
        on="customer_unique_id",
        how="left"
    )

    features_df["customer_display"] = (
        features_df["customer_code"]
        + " | "
        + features_df["customer_city"].astype(str).str.title()
        + " | "
        + features_df["customer_state"].astype(str)
    )

    # ==========================================================
    # PREPARE FEATURES
    # ==========================================================
    X_all = features_df.drop(columns=[
        "customer_unique_id",
        "customer_code",
        "customer_city",
        "customer_state",
        "customer_display"
    ], errors="ignore")

    for col in model_features:
        if col not in X_all.columns:
            X_all[col] = 0

    X_all = X_all[model_features]

    probabilities = model.predict_proba(X_all)[:, 1]
    features_df["retention_probability"] = probabilities

    # ==========================================================
    # 🔥 EXECUTIVE SUMMARY
    # ==========================================================
    st.subheader("Executive Summary")

    avg_retention = features_df["retention_probability"].mean()

    high_risk = (features_df["retention_probability"] < 0.3).sum()
    medium_risk = ((features_df["retention_probability"] >= 0.3) & 
                   (features_df["retention_probability"] < 0.7)).sum()
    low_risk = (features_df["retention_probability"] >= 0.7).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Retention", f"{avg_retention:.2f}")
    col2.metric("High Risk", high_risk)
    col3.metric("Medium Risk", medium_risk)
    col4.metric("Low Risk", low_risk)

    # ==========================================================
    # DISTRIBUTION
    # ==========================================================
    st.subheader("Retention Distribution")
    st.bar_chart(features_df["retention_probability"])

    # ==========================================================
    # TOP CUSTOMERS
    # ==========================================================
    st.subheader("Top Likely To Retain")

    top_customers = features_df.sort_values(
        "retention_probability", ascending=False
    ).head(5)

    st.dataframe(top_customers[["customer_display", "retention_probability"]])

    # ==========================================================
    # HIGH RISK CUSTOMERS
    # ==========================================================
    st.subheader("Customers at Risk")

    risky = features_df.sort_values("retention_probability").head(10)

    st.dataframe(risky[["customer_display", "retention_probability"]])

    st.markdown("---")

    # ==========================================================
    # CUSTOMER SELECTION
    # ==========================================================
    customer_map = dict(
        zip(features_df["customer_display"], features_df["customer_unique_id"])
    )

    selected_display = st.selectbox("Select Customer", list(customer_map.keys()))
    selected_customer = customer_map[selected_display]

    customer_data = features_df[
        features_df["customer_unique_id"] == selected_customer
    ].drop(columns=[
        "customer_unique_id",
        "customer_code",
        "customer_city",
        "customer_state",
        "customer_display",
        "retention_probability"
    ], errors="ignore")

    for col in model_features:
        if col not in customer_data.columns:
            customer_data[col] = 0

    customer_data = customer_data[model_features]

    probability = model.predict_proba(customer_data)[0][1]

    st.subheader("Customer Retention Score")
    st.metric("Score", f"{probability * 100:.2f}")

    # ==========================================================
    # 🔥 RISK LOGIC
    # ==========================================================
    def get_risk(p):
        if p < 0.3:
            return "High Risk", "Immediate retention campaign required"
        elif p < 0.7:
            return "Medium Risk", "Engagement strategy recommended"
        else:
            return "Low Risk", "Customer is stable"

    risk, action = get_risk(probability)

    if risk == "High Risk":
        st.error(f"{risk} 🚨")
    elif risk == "Medium Risk":
        st.warning(f"{risk} ⚠️")
    else:
        st.success(f"{risk} ✅")

    st.info(f"Recommended Action: {action}")

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================
    st.subheader("Key Drivers")

    try:
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": model_features,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))

    except:
        st.info("Feature importance not available")