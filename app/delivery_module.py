def run_delivery_dashboard():

    import streamlit as st
    import pandas as pd
    import joblib
    import plotly.graph_objects as go
    import os
    from datetime import datetime, timedelta

    # ==========================================================
    # TIME (IST)
    # ==========================================================
    current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)

    # ==========================================================
    # SAFE SHAP IMPORT
    # ==========================================================
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False

    # ==========================================================
    # LOAD MODEL (FIXED)
    # ==========================================================
    @st.cache_resource
    def load_assets():
        model_path = "models/delivery_model.pkl"   # ✅ FIXED

        if not os.path.exists(model_path):
            st.error("❌ Delivery model not found. Please train first.")
            st.stop()

        model = joblib.load(model_path)

        try:
            features = joblib.load("models/model_features.pkl")
        except:
            features = None

        return model, features

    model, EXPECTED_FEATURES = load_assets()

    # ==========================================================
    # HEADER
    # ==========================================================
    st.markdown("# 🚚 Delivery Risk Intelligence")
    st.caption("AI-powered delay risk prediction with explainability")
    st.markdown(f"Last Updated: {current_time.strftime('%d %B %Y, %H:%M:%S')}")
    st.markdown("---")

    # ==========================================================
    # INPUT PANEL
    # ==========================================================
    st.markdown("### ⚙️ Risk Control Panel")

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
    }])

    # ==========================================================
    # FEATURE ALIGNMENT
    # ==========================================================
    if isinstance(EXPECTED_FEATURES, list):
        for col in EXPECTED_FEATURES:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[EXPECTED_FEATURES]

    # ==========================================================
    # PREDICTION
    # ==========================================================
    probability = float(model.predict_proba(input_data)[0][1])
    risk_score = probability * 100

    # ==========================================================
    # RISK LABEL (NEW 🔥)
    # ==========================================================
    if risk_score < 30:
        risk_label = "🟢 LOW RISK"
    elif risk_score < 70:
        risk_label = "🟡 MEDIUM RISK"
    else:
        risk_label = "🔴 HIGH RISK"

    # ==========================================================
    # DISPLAY METRICS
    # ==========================================================
    colA, colB = st.columns(2)

    with colA:
        st.metric("Delay Risk (%)", f"{risk_score:.2f}")

    with colB:
        st.metric("Risk Level", risk_label)

    # ==========================================================
    # ANALYSIS BUTTON
    # ==========================================================
    if st.button("🔍 Run Detailed Analysis"):

        # ------------------ Gauge ------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Delivery Delay Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if risk_score > 70 else "orange" if risk_score > 30 else "green"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ------------------ SHAP ------------------
        st.markdown("## 🧠 Risk Driver Analysis")

        if SHAP_AVAILABLE:
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(input_data)

                shap_df = pd.DataFrame({
                    "Feature": input_data.columns,
                    "Impact": shap_values.values[0]
                }).sort_values("Impact", key=abs, ascending=False)

                st.dataframe(shap_df.head(5))

            except Exception as e:
                st.warning(f"SHAP failed: {e}")

        else:
            st.warning("SHAP not installed")

    # ==========================================================
    # INSIGHT BOX (NEW 🔥)
    # ==========================================================
    st.markdown("---")

    if risk_score > 70:
        st.error("⚠️ High delay risk. Immediate action recommended.")
    elif risk_score > 30:
        st.warning("⚠️ Moderate risk. Monitor closely.")
    else:
        st.success("✅ Delivery is expected on time.")