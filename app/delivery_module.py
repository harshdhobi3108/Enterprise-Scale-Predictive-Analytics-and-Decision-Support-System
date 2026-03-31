def run_delivery_dashboard():

    import streamlit as st
    import pandas as pd
    import joblib
    import plotly.graph_objects as go
    from datetime import datetime, timedelta

    # ==========================================================
    # TIME
    # ==========================================================
    current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    model = joblib.load("models/delivery_model.pkl")
    FEATURES = joblib.load("models/model_features.pkl")
    auto_threshold = joblib.load("models/threshold.pkl")

    # 🔥 FINAL SAFE THRESHOLD
    threshold = max(0.3, auto_threshold)

    # ==========================================================
    # HEADER
    # ==========================================================
    st.markdown("# 🚚 Delivery Risk Intelligence")
    st.caption("Enterprise AI-powered delay prediction")
    st.markdown(f"Last Updated: {current_time.strftime('%d %B %Y, %H:%M:%S')}")
    st.markdown("---")

    # ==========================================================
    # INPUT PANEL
    # ==========================================================
    st.markdown("### ⚙️ Risk Control Panel")

    col1, col2 = st.columns(2)

    with col1:
        purchase_hour = st.slider("Hour", 0, 23, 12)
        purchase_dayofweek = st.slider("Day of Week", 0, 6, 2)
        purchase_month = st.slider("Month", 1, 12, 6)

    with col2:
        total_payment_value = st.number_input("Payment Value", 0.0, 10000.0, 150.0)
        payment_installments = st.slider("Installments", 1, 24, 1)

    # ==========================================================
    # INPUT DATA
    # ==========================================================
    input_data = pd.DataFrame([{
        "purchase_hour": purchase_hour,
        "purchase_dayofweek": purchase_dayofweek,
        "purchase_month": purchase_month,
        "total_payment_value": total_payment_value,
        "payment_installments": payment_installments,
    }])

    input_data = input_data[FEATURES]

    # ==========================================================
    # PREDICTION
    # ==========================================================
    probability = float(model.predict_proba(input_data)[0][1])
    risk_score = probability * 100

    prediction = 1 if probability >= threshold else 0

    # ==========================================================
    # CONFIDENCE
    # ==========================================================
    confidence = abs(probability - 0.5) * 2 * 100

    # ==========================================================
    # RISK LABEL
    # ==========================================================
    if probability < 0.3:
        risk_label = "🟢 LOW RISK"
    elif probability < 0.7:
        risk_label = "🟡 MEDIUM RISK"
    else:
        risk_label = "🔴 HIGH RISK"

    # ==========================================================
    # DISPLAY
    # ==========================================================
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Delay Risk (%)", f"{risk_score:.2f}")

    with colB:
        st.metric("Risk Level", risk_label)

    with colC:
        st.metric("Confidence (%)", f"{confidence:.2f}")

    # ==========================================================
    # GAUGE
    # ==========================================================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Delivery Delay Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {
                'color': "red" if probability > 0.7 else "orange" if probability > 0.3 else "green"
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # INSIGHT
    # ==========================================================
    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High probability of delivery delay detected.")
    else:
        st.success("✅ Delivery is likely to be on time.")