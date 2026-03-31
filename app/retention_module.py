def run_retention_dashboard():

    import streamlit as st
    import joblib
    import pandas as pd

    from src.data_loader import DataLoader
    from src.retention_features import build_retention_features
    from src.retention_explainer import RetentionExplainer

    MODEL_PATH = "models/retention_model.pkl"

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    model_bundle = joblib.load(MODEL_PATH)

    model = model_bundle["model"] if isinstance(model_bundle, dict) else model_bundle
    threshold = model_bundle.get("threshold", 0.5) if isinstance(model_bundle, dict) else 0.5

    explainer = RetentionExplainer(MODEL_PATH)

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
    # HEADER
    # ==========================================================
    st.markdown("## 🔁 Customer Retention Intelligence")
    st.caption("Predict churn and understand drivers")
    st.markdown("---")

    # ==========================================================
    # SELECT CUSTOMER
    # ==========================================================
    selected_index = st.selectbox("Select Customer Index", df.index)

    customer = df.iloc[[selected_index]]

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
    # SHAP
    # ==========================================================
    st.markdown("### 🧠 Model Explainability")

    try:
        shap_values, X = explainer.explain_instance(customer)
        st.write("Top influencing features shown below")
        st.dataframe(X.head())
    except:
        st.info("Explainability not available")