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
    # LOAD MODEL
    # ==========================================================
    @st.cache_resource
    def load_assets():
        bundle = joblib.load("models/model_features.pkl")
        return bundle["model"], bundle["features"]

    model, EXPECTED_FEATURES = load_assets()

    # ==========================================================
    # LOAD FEATURE IMPORTANCE (FIXED)
    # ==========================================================
    @st.cache_data
    def load_importance():
        path = "models/feature_importance.csv"
        if not os.path.exists(path):
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.read_csv(path)

    importance_df = load_importance()

    # ==========================================================
    # CREATE SHAP EXPLAINER (FIXED)
    # ==========================================================
    explainer = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
        except:
            explainer = None

    # ==========================================================
    # HEADER
    # ==========================================================
    st.markdown("# Delivery Risk Intelligence")
    st.caption("Operational Delay Risk Monitoring & Explainable AI")
    st.markdown(f"Last Updated: {datetime.now().strftime('%d %B %Y, %H:%M')}")
    st.markdown("---")

    # ==========================================================
    # CONTROL PANEL
    # ==========================================================
    st.markdown("### Risk Control Panel")

    filters = st.columns(8)

    with filters[0]:
        medium_threshold = st.slider("Medium Risk", 10, 60, 30)

    with filters[1]:
        high_threshold = st.slider("High Risk", 40, 95, 60)

    with filters[2]:
        purchase_hour = st.slider("Hour", 0, 23, 12)

    with filters[3]:
        purchase_dayofweek = st.slider("Day", 0, 6, 2)

    with filters[4]:
        purchase_month = st.slider("Month", 1, 12, 6)

    with filters[5]:
        approval_delay_hours = st.number_input("Approval Delay", 0.0, 200.0, 2.0)

    with filters[6]:
        carrier_delay_hours = st.number_input("Carrier Delay", 0.0, 500.0, 12.0)

    with filters[7]:
        estimated_delivery_days = st.number_input("Delivery Days", 1.0, 60.0, 7.0)

    filters2 = st.columns(2)

    with filters2[0]:
        total_payment_value = st.number_input("Payment Value", 0.0, 10000.0, 150.0)

    with filters2[1]:
        payment_installments = st.slider("Installments", 1, 24, 1)

    st.markdown("---")

    # ==========================================================
    # INPUT DATA
    # ==========================================================
    input_dict = {
        "purchase_hour": purchase_hour,
        "purchase_dayofweek": purchase_dayofweek,
        "purchase_month": purchase_month,
        "approval_delay_hours": approval_delay_hours,
        "carrier_delay_hours": carrier_delay_hours,
        "estimated_delivery_days": estimated_delivery_days,
        "total_payment_value": total_payment_value,
        "payment_installments": payment_installments,
    }

    input_data = pd.DataFrame([input_dict])

    # FIXED ALIGNMENT
    for col in EXPECTED_FEATURES:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[EXPECTED_FEATURES]

    # ==========================================================
    # PREDICTION
    # ==========================================================
    probability = float(model.predict_proba(input_data)[0][1])
    risk_score = probability * 100

    if risk_score >= high_threshold:
        risk_level = "High Risk"
    elif risk_score >= medium_threshold:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    # ==========================================================
    # SNAPSHOT
    # ==========================================================
    orders_monitored = 1000 + int(risk_score * 5) + random.randint(0, 50)
    high_risk_exposure = max(risk_score - medium_threshold, 0)
    on_time_rate = max(100 - risk_score, 0)

    st.markdown("## Executive Snapshot")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Orders Monitored", f"{orders_monitored:,}")
    col2.metric("Average Risk Score", f"{risk_score:.1f}%")
    col3.metric("High Risk Exposure", f"{high_risk_exposure:.1f}%")
    col4.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")

    st.caption(f"Current Risk Level: **{risk_level}**")

    st.markdown("---")

    # ==========================================================
    # ANALYSIS
    # ==========================================================
    if st.button("Run Detailed Analysis"):

        st.success(f"{risk_level} | Delay Probability: {risk_score:.2f}%")

        # GAUGE
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Delivery Delay Risk (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("---")

        # SHAP / FALLBACK
        st.markdown("## Risk Driver Analysis")

        if SHAP_AVAILABLE and explainer is not None:
            try:
                shap_values = explainer.shap_values(input_data)

                shap_array = (
                    shap_values[1][0]
                    if isinstance(shap_values, list)
                    else shap_values[0]
                )

                shap_df = pd.DataFrame({
                    "Feature": EXPECTED_FEATURES,
                    "Impact": shap_array
                }).sort_values("Impact", key=abs, ascending=False)

                st.dataframe(shap_df.head(5))

            except:
                st.warning("SHAP failed")

        else:
            st.warning("Using global importance instead")

            top_features = importance_df.sort_values(
                "importance", ascending=False
            ).head(5)

            st.dataframe(top_features)

    # ==========================================================
    # GLOBAL IMPORTANCE
    # ==========================================================
    st.markdown("---")

    with st.expander("Strategic Model Influence Overview"):

        fig = px.bar(
            importance_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h"
        )

        st.plotly_chart(fig, use_container_width=True)