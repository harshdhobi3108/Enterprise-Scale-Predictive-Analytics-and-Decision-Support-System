def run_delivery_dashboard():

    import streamlit as st
    import pandas as pd
    import joblib
    import plotly.graph_objects as go
    import plotly.express as px
    import shap
    from datetime import datetime

    # ==========================================================
    # MODEL LOADING (SAFE + CACHED)
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
    # SIDEBAR CONTROLS
    # ==========================================================
    st.sidebar.markdown("## Risk Control Panel")

    medium_threshold = st.sidebar.slider(
        "Medium Risk Threshold (%)", 10, 60, 30, 5
    )

    high_threshold = st.sidebar.slider(
        "High Risk Threshold (%)", 40, 95, 60, 5
    )

    st.sidebar.markdown("---")

    purchase_hour = st.sidebar.slider("Purchase Hour", 0, 23, 12)
    purchase_dayofweek = st.sidebar.slider("Day of Week", 0, 6, 2)
    purchase_month = st.sidebar.slider("Purchase Month", 1, 12, 6)

    approval_delay_hours = st.sidebar.number_input(
        "Approval Delay (Hours)", 0.0, 200.0, 2.0
    )

    carrier_delay_hours = st.sidebar.number_input(
        "Carrier Delay (Hours)", 0.0, 500.0, 12.0
    )

    estimated_delivery_days = st.sidebar.number_input(
        "Estimated Delivery Days", 1.0, 60.0, 7.0
    )

    total_payment_value = st.sidebar.number_input(
        "Payment Value", 0.0, 10000.0, 150.0
    )

    payment_installments = st.sidebar.slider("Installments", 1, 24, 1)

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
    # EXECUTIVE SNAPSHOT
    # ==========================================================
    st.markdown("## Executive Snapshot")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Orders Monitored", "1,250")
    col2.metric("Average Risk Score", "42%")
    col3.metric("High Risk Exposure", "28%")
    col4.metric("On-Time Delivery Rate", "72%")

    st.markdown("---")

    # ==========================================================
    # RUN PREDICTION
    # ==========================================================
    if st.button("Run Risk Assessment"):

        probability = float(model.predict_proba(input_data)[0][1])
        risk_score = probability * 100

        # Dynamic Classification
        if risk_score >= high_threshold:
            status = "High Risk — Immediate Operational Intervention Required"
        elif risk_score >= medium_threshold:
            status = "Moderate Risk — Enhanced Monitoring Recommended"
        else:
            status = "Low Risk — Operations Within Acceptable Range"

        st.markdown(
            f"""
            <div style='
                padding:18px;
                border-radius:10px;
                background-color: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                font-weight:600;
                text-align:center;'>
                {status}
                <br>
                Predicted Delay Probability: {risk_score:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ======================================================
        # GAUGE VISUALIZATION
        # ======================================================
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Predicted Delay Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6"},
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("---")

        # ======================================================
        # SHAP ANALYSIS
        # ======================================================
        st.markdown("## Risk Driver Analysis")

        shap_values = explainer.shap_values(input_data)
        shap_array = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": EXPECTED_FEATURES,
            "Impact": shap_array
        })

        shap_df["abs_impact"] = shap_df["Impact"].abs()
        shap_df = shap_df.sort_values("abs_impact", ascending=False)

        # Business-friendly naming
        feature_map = {
            "approval_delay_hours": "Approval Processing Delay",
            "carrier_delay_hours": "Carrier Delay",
            "estimated_delivery_days": "Delivery Window Duration",
            "total_payment_value": "Transaction Value",
            "payment_installments": "Installment Count",
            "purchase_hour": "Purchase Hour",
            "purchase_month": "Purchase Month",
            "purchase_dayofweek": "Day of Week"
        }

        shap_df["Feature"] = shap_df["Feature"].map(
            lambda x: feature_map.get(x, x)
        )

        top3 = shap_df.head(3)

        fig = px.bar(
            top3.sort_values("Impact"),
            x="Impact",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Executive Interpretation
        st.markdown("### Executive Interpretation")

        for _, row in top3.iterrows():
            direction = "increasing" if row["Impact"] > 0 else "reducing"
            st.write(
                f"- {row['Feature']} is currently {direction} overall delay probability."
            )

    # ==========================================================
    # GLOBAL MODEL INTELLIGENCE
    # ==========================================================
    st.markdown("---")

    with st.expander("Strategic Model Influence Overview"):

        fig_imp = px.bar(
            importance_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h"
        )

        st.plotly_chart(fig_imp, use_container_width=True)