def run_delivery_dashboard():

    import streamlit as st
    import pandas as pd
    import joblib
    import plotly.graph_objects as go
    import plotly.express as px
    import shap

    # ==========================================================
    # ULTRA MODERN DARK THEME
    # ==========================================================
    st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */
.stApp {
    background: radial-gradient(circle at top left, #0f172a, #050816 70%);
}

/* ===== SIDEBAR BASE ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220, #050816);
    border-right: 1px solid rgba(0, 255, 255, 0.15);
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.05);
    padding-top: 25px;
}

/* ===== SIDEBAR TITLE ===== */
section[data-testid="stSidebar"] h1 {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #00f0ff !important;
    letter-spacing: 1px;
}

/* ===== SIDEBAR SECTION HEADERS ===== */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6ee7ff !important;
    margin-top: 25px;
}

/* ===== SLIDER TRACK ===== */
section[data-testid="stSidebar"] .stSlider > div > div {
    color: #00f0ff;
}

/* ===== BUTTON STYLE ===== */
.stButton > button {
    background: linear-gradient(90deg, #00f0ff, #7c3aed);
    color: black;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
}

.stButton > button:hover {
    box-shadow: 0 0 15px #00f0ff;
    transition: 0.3s;
}

/* ===== KPI CARDS ===== */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(0,255,255,0.1);
    padding: 28px;
    border-radius: 18px;
    backdrop-filter: blur(8px);
}

/* ===== BLOCK CONTAINER ===== */
.block-container {
    max-width: 1500px;
}

</style>
""", unsafe_allow_html=True)

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    @st.cache_resource
    def load_model():
        model = joblib.load("models/delivery_delay_model.pkl")
        explainer = shap.TreeExplainer(model)
        importance_df = pd.read_csv("models/feature_importance.csv")
        return model, explainer, importance_df

    model, explainer, importance_df = load_model()
    EXPECTED_FEATURES = list(model.feature_name_)

    # ==========================================================
    # HEADER
    # ==========================================================
    st.markdown("""
<style>

/* ===== HERO TITLE GLOW ===== */
.hero-title {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #00f0ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ===== SUBTITLE ===== */
.hero-subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 8px;
}

/* ===== STATUS BADGE ===== */
.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    background: rgba(0,255,255,0.1);
    border: 1px solid rgba(0,255,255,0.3);
    font-size: 13px;
    color: #00f0ff;
    margin-top: 15px;
}

/* ===== GLOW LINE ===== */
.glow-line {
    height: 2px;
    background: linear-gradient(90deg, #00f0ff, transparent);
    margin-top: 20px;
    margin-bottom: 25px;
}

</style>

<div>
    <div class="hero-title">Delivery Risk Command Center</div>
    <div class="hero-subtitle">
        Real-Time Operational Risk Monitoring • Predictive AI • Explainability Engine
    </div>
    <div class="status-badge">
        ● System Operational
    </div>
    <div class="glow-line"></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid #1f2937;'>", unsafe_allow_html=True)

   # ==========================================================
    # SIDEBAR INPUT PANEL
    # ==========================================================
    st.sidebar.markdown("## ⚙ CONTROL PANEL")
    st.sidebar.markdown("---")

    # Risk Threshold Controls
    st.sidebar.markdown("### 🎯 Risk Thresholds")
    medium_threshold = st.sidebar.slider(
        "Medium Risk Threshold (%)",
        10, 60, 30, 5
    )

    high_threshold = st.sidebar.slider(
        "High Risk Threshold (%)",
        40, 95, 60, 5
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("### ⏱ Time Parameters")
    purchase_hour = st.sidebar.slider("Purchase Hour", 0, 23, 12)
    purchase_dayofweek = st.sidebar.slider("Day of Week", 0, 6, 2)
    purchase_month = st.sidebar.slider("Purchase Month", 1, 12, 6)

    st.sidebar.markdown("### 🚚 Logistics Parameters")
    approval_delay_hours = st.sidebar.number_input("Approval Delay", 0.0, 200.0, 2.0)
    carrier_delay_hours = st.sidebar.number_input("Carrier Delay", 0.0, 500.0, 12.0)
    estimated_delivery_days = st.sidebar.number_input("Estimated Delivery Days", 1.0, 60.0, 7.0)

    st.sidebar.markdown("### 💳 Financial Parameters")
    total_payment_value = st.sidebar.number_input("Payment Value", 0.0, 10000.0, 150.0)
    payment_installments = st.sidebar.slider("Installments", 1, 24, 1)


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
    # PREDICTION BUTTON
    # ==========================================================
    if st.button("🚀 Run Risk Analysis"):

        probability = float(model.predict_proba(input_data)[0][1])
        risk_score = probability * 100

        # ======================================================
        # RISK STATUS BANNER
        # ======================================================
        if risk_score >= 60:
            banner_color = "#7f1d1d"
            status = "HIGH RISK — Immediate Action Required"
        elif risk_score >= 30:
            banner_color = "#78350f"
            status = "MODERATE RISK — Monitor Closely"
        else:
            banner_color = "#064e3b"
            status = "LOW RISK — Operations Stable"

        st.markdown(
            f"<div class='risk-banner' style='background:{banner_color};'>{status}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ======================================================
        # GAUGE VISUALIZATION
        # ======================================================
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Delay Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 30], 'color': "#065f46"},
                    {'range': [30, 60], 'color': "#78350f"},
                    {'range': [60, 100], 'color': "#7f1d1d"}
                ],
            }
        ))

        gauge.update_layout(
            paper_bgcolor="#0a0f1c",
            font={'color': "white"}
        )

        st.plotly_chart(gauge, width="stretch")

        st.markdown("<hr style='border:1px solid #1f2937;'>", unsafe_allow_html=True)

        # ======================================================
        # SHAP FEATURE IMPACT
        # ======================================================
        st.subheader("AI Feature Impact Analysis")

        shap_values = explainer.shap_values(input_data)
        shap_array = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": EXPECTED_FEATURES,
            "Impact": shap_array
        }).sort_values("Impact")

        fig = px.bar(
            shap_df,
            x="Impact",
            y="Feature",
            orientation="h",
            template="plotly_dark"
        )

        fig.update_layout(
            plot_bgcolor="#0a0f1c",
            paper_bgcolor="#0a0f1c",
            font=dict(color="white")
        )

        st.plotly_chart(fig, width="stretch")

    st.markdown("<hr style='border:1px solid #1f2937;'>", unsafe_allow_html=True)

    # ==========================================================
    # GLOBAL MODEL INSIGHTS
    # ==========================================================
    st.subheader("Global Model Intelligence")

    fig_imp = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        template="plotly_dark"
    )

    fig_imp.update_layout(
        plot_bgcolor="#0a0f1c",
        paper_bgcolor="#0a0f1c",
        font=dict(color="white")
    )

    st.plotly_chart(fig_imp, width="stretch")
