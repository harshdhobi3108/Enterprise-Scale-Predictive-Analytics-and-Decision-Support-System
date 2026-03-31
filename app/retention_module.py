def run_retention_dashboard():

    import streamlit as st
    import joblib
    import pandas as pd
    import os
    import random

    from src.data_loader import DataLoader
    from src.retention_features import build_retention_features

    # ==========================================================
    # MODEL PATH (SAFE)
    # ==========================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "retention_model.pkl")

    # ==========================================================
    # HEADER
    # ==========================================================
    st.markdown("## 🔁 Customer Retention Intelligence")
    st.caption("Predict churn and understand customer behavior")
    st.markdown("---")

    # ==========================================================
    # LOAD DATA
    # ==========================================================
    @st.cache_data
    def load_data():
        loader = DataLoader("data/raw")
        data = loader.load_all()

        features = build_retention_features(
            data["orders"],
            data["payments"],
            data["reviews"]
        )

        return features

    df = load_data()

    # ==========================================================
    # SELECT CUSTOMER
    # ==========================================================
    selected_index = st.selectbox("Select Customer Index", df.index)
    customer = df.iloc[[selected_index]]

    # ==========================================================
    # CHECK MODEL
    # ==========================================================
    if not os.path.exists(MODEL_PATH):

        # 🚀 DEMO MODE (NO CRASH)
        st.warning("⚠️ Retention model not found — running in demo mode")

        prob = random.uniform(0.3, 0.9)

        col1, col2 = st.columns(2)

        col1.metric("Retention Probability", f"{prob*100:.2f}%")
        col2.metric("Status", "Retained" if prob > 0.5 else "Churn Risk")

        st.progress(prob)

        st.markdown("### 📊 Insights")
        if prob < 0.4:
            st.warning("- Customer shows high churn tendency")
        elif prob > 0.7:
            st.success("- Customer is highly loyal")
        else:
            st.info("- Moderate retention probability")

        st.info("ℹ️ This is simulated output because model is missing")

        return  # ⛔ STOP HERE (no crash)

    # ==========================================================
    # LOAD MODEL (REAL MODE)
    # ==========================================================
    model_bundle = joblib.load(MODEL_PATH)

    if isinstance(model_bundle, dict):
        model = model_bundle["model"]
        threshold = model_bundle.get("threshold", 0.5)
    else:
        model = model_bundle
        threshold = 0.5

    # ==========================================================
    # PREDICTION
    # ==========================================================
    prob = model.predict_proba(customer)[0][1]

    col1, col2 = st.columns(2)

    col1.metric("Retention Probability", f"{prob*100:.2f}%")
    col2.metric("Status", "Retained" if prob >= threshold else "Churn Risk")

    # ==========================================================
    # PROGRESS BAR
    # ==========================================================
    st.progress(prob)

    # ==========================================================
    # INSIGHTS
    # ==========================================================
    st.markdown("### 📊 Insights")

    insights = []

    if prob < 0.4:
        insights.append("Customer has high churn probability")

    if prob > 0.7:
        insights.append("Customer is highly loyal")

    if not insights:
        insights.append("Moderate retention likelihood")

    for i in insights:
        st.warning(f"- {i}")

    # ==========================================================
    # OPTIONAL EXPLAINABILITY (SAFE)
    # ==========================================================
    st.markdown("### 🧠 Model Explainability")

    try:
        from src.retention_explainer import RetentionExplainer

        explainer = RetentionExplainer(MODEL_PATH)
        shap_values, X = explainer.explain_instance(customer)

        st.write("Top influencing features:")
        st.dataframe(X.head())

    except Exception as e:
        st.info("Explainability not available")