def run_delivery_dashboard():

    import streamlit as st
    import pandas as pd
    import joblib
    import plotly.graph_objects as go
    import plotly.express as px
    import random
    import os
    from datetime import datetime

    # ==========================================================
    # SAFE SHAP IMPORT
    # ==========================================================
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False

    # ==========================================================
    # LOAD MODEL + FEATURES (FIXED ONLY)
    # ==========================================================
    @st.cache_resource
    def load_assets():

        model = joblib.load("models/retention_model.pkl")

        try:
            # ✅ FIXED: load correct features file
            features = joblib.load("models/model_features.pkl")
        except:
            features = None

        return model, features

    model, EXPECTED_FEATURES = load_assets()

    # ==========================================================
    # LOAD FEATURE IMPORTANCE (UNCHANGED)
    # ==========================================================
    @st.cache_data
    def load_importance():
        path = "models/feature_importance.csv"
        if not os.path.exists(path):
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.read_csv(path)

    importance_df = load_importance()

    # ==========================================================
    # SHAP EXPLAINER (UNCHANGED)
    # ==========================================================
    explainer = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
        except:
            explainer = None

    # ==========================================================
    # HEADER (UNCHANGED)
    # ==========================================================
    st.markdown("# Delivery Risk Intelligence")
    st.caption("Operational Delay Risk Monitoring & Explainable AI")
    st.markdown(f"Last Updated: {datetime.now().strftime('%d %B %Y, %H:%M')}")
    st.markdown("---")

    # ==========================================================
    # INPUT PANEL (UNCHANGED)
    # ==========================================================
    st.markdown("### Risk Control Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        purchase_hour = st.slider("Hour", 0, 23, 12)
        purchase_dayofweek = st.slider("Day", 0, 6, 2)
        purchase_month = st.slider("Month", 1, 12, 6)

    with col2:
        approval_delay_hours = st.number_input("Approval Delay", 0.0, 200.0, 2.0)
        carrier_delay_hours = st.number_input("Carrier Delay", 0.0, 500.0, 12.0)
        estimated_delivery_days = st.number_input("Delivery Days", 1.0, 60.0, 7.0)

    with col3:
        total_payment_value = st.number_input("Payment Value", 0.0, 10000.0, 150.0)
        payment_installments = st.slider("Installments", 1, 24, 1)

    # ==========================================================
    # INPUT DATA (UNCHANGED)
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

    # ==========================================================
    # SAFE FEATURE ALIGNMENT (MINIMAL FIX)
    # ==========================================================
    if isinstance(EXPECTED_FEATURES, list):
        for col in EXPECTED_FEATURES:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[EXPECTED_FEATURES]

    # ==========================================================
    # PREDICTION (UNCHANGED)
    # ==========================================================
    probability = float(model.predict_proba(input_data)[0][1])
    risk_score = probability * 100

    st.metric("Delay Risk (%)", f"{risk_score:.2f}")

    # ==========================================================
    # ANALYSIS BUTTON (UNCHANGED)
    # ==========================================================
    if st.button("Run Detailed Analysis"):

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Delivery Delay Risk (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Risk Driver Analysis")

        # ======================================================
        # SHAP (UNCHANGED)
        # ======================================================
        if SHAP_AVAILABLE and explainer is not None:
            try:
                shap_values = explainer.shap_values(input_data)
                shap_array = shap_values[1][0]

                shap_df = pd.DataFrame({
                    "Feature": input_data.columns,
                    "Impact": shap_array
                }).sort_values("Impact", key=abs, ascending=False)

                st.dataframe(shap_df.head(5))

            except Exception as e:
                st.warning(f"SHAP failed: {e}")

        else:
            st.warning("Using global importance")

            st.dataframe(
                importance_df.sort_values("importance", ascending=False).head(5)
            )
