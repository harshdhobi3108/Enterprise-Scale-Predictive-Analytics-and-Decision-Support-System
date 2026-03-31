def run_delivery_dashboard():

    import streamlit as st
    import pandas as pd
    import joblib
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

    threshold = max(0.3, auto_threshold)

    # ==========================================================
    # STYLING (ENTERPRISE UI)
    # ==========================================================
    st.markdown("""
    <style>
    .card {
        background: #111827;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
    }
    .title {
        font-size: 14px;
        color: #9ca3af;
    }
    .value {
        font-size: 32px;
        font-weight: bold;
    }
    .high { color: #ef4444; }
    .medium { color: #f59e0b; }
    .low { color: #22c55e; }
    </style>
    """, unsafe_allow_html=True)

    # ==========================================================
    # HEADER
    # ==========================================================
    st.markdown("## 🚚 Delivery Risk Intelligence")
    st.caption("AI-powered delay prediction")
    st.write(f"Last Updated: {current_time.strftime('%d %B %Y, %H:%M:%S')}")
    st.markdown("---")

    # ==========================================================
    # INPUT PANEL (CLEAN CARD)
    # ==========================================================
    st.markdown("### ⚙️ Risk Configuration")

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
    confidence = abs(probability - 0.5) * 2 * 100

    # ==========================================================
    # RISK LABEL
    # ==========================================================
    if probability < 0.3:
        risk_label = "LOW"
        risk_class = "low"
    elif probability < 0.7:
        risk_label = "MEDIUM"
        risk_class = "medium"
    else:
        risk_label = "HIGH"
        risk_class = "high"

    # ==========================================================
    # KPI CARDS (🔥 BIG UPGRADE)
    # ==========================================================
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown(f"""
        <div class="card">
            <div class="title">Delay Risk</div>
            <div class="value">{risk_score:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class="card">
            <div class="title">Risk Level</div>
            <div class="value {risk_class}">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class="card">
            <div class="title">Confidence</div>
            <div class="value">{confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================
    # PROGRESS BAR (REPLACES GAUGE)
    # ==========================================================
    st.markdown("### 📊 Delivery Risk Score")
    st.progress(min(max(probability, 0.0), 1.0))
    st.write(f"**{risk_score:.2f}% Risk Probability**")

    # ==========================================================
    # SMART INSIGHTS (🔥 THIS MAKES IT ENTERPRISE)
    # ==========================================================
    st.markdown("### 📊 Risk Insights")

    insights = []

    if purchase_hour in [18, 19, 20, 21]:
        insights.append("Peak hour traffic may cause delays")

    if payment_installments <= 2:
        insights.append("Low installment orders show higher risk pattern")

    if total_payment_value > 500:
        insights.append("High transaction value increases processing risk")

    if purchase_dayofweek in [5, 6]:
        insights.append("Weekend deliveries have higher delay probability")

    if not insights:
        insights.append("No strong risk indicators detected")

    for i in insights:
        st.warning(f"- {i}")

    # ==========================================================
    # ACTIONABLE RECOMMENDATIONS
    # ==========================================================
    st.markdown("### 🚀 Recommended Actions")

    if prediction == 1:
        st.error("""
        - Prioritize this order for dispatch  
        - Enable real-time tracking  
        - Consider alternative delivery routes  
        - Flag for manual review  
        """)
    else:
        st.success("""
        - Proceed with standard delivery  
        - No intervention required  
        """)
