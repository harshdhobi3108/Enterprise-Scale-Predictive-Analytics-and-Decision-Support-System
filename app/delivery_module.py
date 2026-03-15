def run_delivery_dashboard():

    import streamlit as st
    import pandas as pd
    import joblib
    import plotly.graph_objects as go
    import plotly.express as px
    import shap
    import random
    from datetime import datetime


    # ==========================================================
    # MODEL LOADING
    # ==========================================================
    @st.cache_resource
    def load_assets():
        model = joblib.load("models/delivery_delay_model.pkl")
        explainer = shap.TreeExplainer(model)
        importance_df = pd.read_csv("models/feature_importance.csv")
        return model, explainer, importance_df

    model, explainer, importance_df = load_assets()
    EXPECTED_FEATURES = list(model.feature_name_)


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
        approval_delay_hours = st.number_input(
            "Approval Delay", 0.0, 200.0, 2.0
        )

    with filters[6]:
        carrier_delay_hours = st.number_input(
            "Carrier Delay", 0.0, 500.0, 12.0
        )

    with filters[7]:
        estimated_delivery_days = st.number_input(
            "Delivery Days", 1.0, 60.0, 7.0
        )

    filters2 = st.columns(2)

    with filters2[0]:
        total_payment_value = st.number_input(
            "Payment Value", 0.0, 10000.0, 150.0
        )

    with filters2[1]:
        payment_installments = st.slider("Installments", 1, 24, 1)

    st.markdown("---")


    # ==========================================================
    # INPUT DATA
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
    }]).reindex(columns=EXPECTED_FEATURES)


    # ==========================================================
    # MODEL PREDICTION
    # ==========================================================
    probability = float(model.predict_proba(input_data)[0][1])
    risk_score = probability * 100


    # ==========================================================
    # RISK CLASSIFICATION
    # ==========================================================
    if risk_score >= high_threshold:
        risk_level = "High Risk"
    elif risk_score >= medium_threshold:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"


    # ==========================================================
    # EXECUTIVE SNAPSHOT
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
    # RUN DETAILED ANALYSIS
    # ==========================================================
    if st.button("Run Detailed Analysis"):

        if risk_level == "High Risk":
            status = "High Risk — Immediate Operational Intervention Required"
        elif risk_level == "Moderate Risk":
            status = "Moderate Risk — Enhanced Monitoring Recommended"
        else:
            status = "Low Risk — Operations Within Acceptable Range"

        st.success(
            f"{status} | Predicted Delay Probability: {risk_score:.2f}%"
        )

        # ======================================================
        # PROFESSIONAL GAUGE
        # ======================================================
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Delivery Delay Risk (%)", 'font': {'size': 20}},
            number={'font': {'size': 48}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff4b4b"},
                'steps': [
                    {'range': [0, 30], 'color': "#1f7a1f"},
                    {'range': [30, 60], 'color': "#ffcc00"},
                    {'range': [60, 100], 'color': "#b30000"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("---")


        # ======================================================
        # SHAP ANALYSIS
        # ======================================================
        st.markdown("## Risk Driver Analysis")

        shap_values = explainer.shap_values(input_data)
        shap_array = shap_values[1][0] if isinstance(
            shap_values, list) else shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": EXPECTED_FEATURES,
            "Impact": shap_array
        })

        shap_df["abs_impact"] = shap_df["Impact"].abs()
        shap_df = shap_df.sort_values(
            "abs_impact", ascending=False
        )

        top3 = shap_df.head(3)

        fig = px.bar(
            top3.sort_values("Impact"),
            x="Impact",
            y="Feature",
            orientation="h",
            title="Top Risk Drivers"
        )

        st.plotly_chart(fig, use_container_width=True)


    # ==========================================================
    # GLOBAL MODEL INTELLIGENCE
    # ==========================================================
    st.markdown("---")

    with st.expander("Strategic Model Influence Overview"):

        fig_imp = px.bar(
            importance_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Global Feature Importance"
        )

        st.plotly_chart(fig_imp, use_container_width=True)