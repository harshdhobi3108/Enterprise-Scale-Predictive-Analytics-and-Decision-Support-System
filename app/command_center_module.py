def run_command_center():

    import streamlit as st
    import plotly.express as px
    import pandas as pd
    from datetime import datetime, timedelta

    current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)

    # ==========================================================
    # LOAD DATA
    # ==========================================================
    @st.cache_data
    def load_processed_data():
        return pd.read_csv("data/processed/delivery_features.csv")

    df = load_processed_data()

    # ==========================================================
    # HEADER
    # ==========================================================
    st.markdown("## 🧠 Enterprise Command Center")
    st.caption("AI-Driven Business Intelligence Overview")
    st.write(f"Last Updated: {current_time.strftime('%d %B %Y | %H:%M')}")
    st.markdown("---")

    # ==========================================================
    # KPI CALCULATIONS
    # ==========================================================
    delivery_risk = df["is_delayed"].mean() * 100

    revenue_risk = (
        (df["total_payment_value"] < df["total_payment_value"].median()).mean() * 100
    )

    churn_risk = (
        (df["payment_installments"] == 1).mean() * 100
    )

    health_score = 100 - int(
        (delivery_risk + revenue_risk + churn_risk) / 3
    )

    # ==========================================================
    # KPI CARDS (🔥 UPGRADE)
    # ==========================================================
    col1, col2, col3, col4 = st.columns(4)

    def card(title, value):
        return f"""
        <div style="
            background:#111827;
            padding:18px;
            border-radius:14px;
            border:1px solid rgba(255,255,255,0.05);
        ">
            <div style="color:#9ca3af;font-size:13px">{title}</div>
            <div style="font-size:28px;font-weight:600">{value}</div>
        </div>
        """

    col1.markdown(card("Delivery Risk", f"{delivery_risk:.1f}%"), unsafe_allow_html=True)
    col2.markdown(card("Revenue Risk", f"{revenue_risk:.1f}%"), unsafe_allow_html=True)
    col3.markdown(card("Churn Risk", f"{churn_risk:.1f}%"), unsafe_allow_html=True)
    col4.markdown(card("Health Score", f"{health_score}/100"), unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================================
    # BAR CHART (REPLACES PIE)
    # ==========================================================
    st.markdown("### 📊 Risk Distribution")

    risk_df = pd.DataFrame({
        "Risk Type": ["Delivery", "Revenue", "Churn"],
        "Risk %": [delivery_risk, revenue_risk, churn_risk]
    })

    fig = px.bar(
        risk_df,
        x="Risk Type",
        y="Risk %",
        text="Risk %",
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        margin=dict(t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # REAL TREND (NOT FAKE)
    # ==========================================================
    st.markdown("### 📈 Delivery Delay Trend")

    trend = df.groupby(df.index // 50)["is_delayed"].mean() * 100
    trend_df = trend.reset_index()
    trend_df.columns = ["Batch", "Delay %"]

    fig_trend = px.line(trend_df, x="Batch", y="Delay %")

    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # ==========================================================
    # EXECUTIVE INSIGHTS
    # ==========================================================
    st.markdown("### 🧠 Executive Insights")

    insights = []

    if delivery_risk > 40:
        insights.append("Delivery operations are under stress")

    if revenue_risk > 40:
        insights.append("Revenue inconsistency detected")

    if churn_risk > 40:
        insights.append("Customer churn risk is elevated")

    if not insights:
        insights.append("System operating within normal parameters")

    for i in insights:
        st.warning(f"- {i}")

    # ==========================================================
    # ACTIONS (🔥 THIS IS WHAT MAKES IT ELITE)
    # ==========================================================
    st.markdown("### 🚀 Recommended Actions")

    if health_score < 60:
        st.error("""
        - Immediate operational review required  
        - Optimize logistics pipeline  
        - Focus on high-risk customers  
        """)
    elif health_score < 75:
        st.info("""
        - Monitor risk segments closely  
        - Improve delivery efficiency  
        """)
    else:
        st.success("""
        - System performing optimally  
        - Maintain current strategy  
        """)