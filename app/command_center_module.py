"""
Enterprise Command Center
Global Business Intelligence Overview
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime


def run_command_center():

    # ==========================================================
    # HEADER
    # ==========================================================

    st.markdown("# Enterprise Command Center")
    st.caption("Unified AI-Driven Business Intelligence Overview")
    st.markdown(f"Last Updated: {datetime.now().strftime('%d %B %Y, %H:%M')}")
    st.markdown("---")

    # ==========================================================
    # ENTERPRISE KPI SIMULATION
    # (Replace later with real model outputs)
    # ==========================================================

    delivery_risk = 42
    revenue_risk = 35
    churn_risk = 28

    health_score = 100 - int(
        (delivery_risk + revenue_risk + churn_risk) / 3
    )

    # ==========================================================
    # EXECUTIVE OVERVIEW
    # ==========================================================

    st.markdown("## Enterprise Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Delivery Risk Exposure", f"{delivery_risk}%")
    col2.metric("Revenue Risk Exposure", f"{revenue_risk}%")
    col3.metric("Churn Risk Exposure", f"{churn_risk}%")
    col4.metric("Business Health Score", f"{health_score}/100")

    st.markdown("---")

    # ==========================================================
    # ENTERPRISE RISK DISTRIBUTION
    # ==========================================================

    st.markdown("## Enterprise Risk Distribution")

    risk_df = pd.DataFrame({
        "Risk Area": ["Delivery", "Revenue", "Churn"],
        "Risk %": [delivery_risk, revenue_risk, churn_risk]
    })

    fig = px.pie(
        risk_df,
        names="Risk Area",
        values="Risk %",
        hole=0.55,
        color_discrete_sequence=["#4F81BD", "#C0504D", "#9BBB59"]
    )

    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        legend_title="Risk Domains"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================================
    # ENTERPRISE STABILITY TREND
    # ==========================================================

    st.markdown("## Enterprise Stability Trend")

    dates = pd.date_range(end=datetime.today(), periods=30)

    base_score = 70
    noise = np.random.normal(0, 3, 30)
    health_trend = np.clip(base_score + noise, 60, 85)

    trend_df = pd.DataFrame({
        "Date": dates,
        "Business Health Score": health_trend
    })

    fig_trend = px.line(
        trend_df,
        x="Date",
        y="Business Health Score"
    )

    fig_trend.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis_range=[55, 90]
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # ==========================================================
    # EXECUTIVE INSIGHT SUMMARY
    # ==========================================================

    st.markdown("## Executive Insight Summary")

    if delivery_risk > 40:
        st.warning(
            "Delivery risk is elevated. Operational logistics optimization recommended."
        )

    if revenue_risk > 40:
        st.warning(
            "Revenue volatility detected. Monitor high-value customer segments."
        )

    if churn_risk > 40:
        st.warning(
            "Customer churn exposure rising. Initiate retention campaigns."
        )

    if health_score >= 75:
        st.success("Enterprise performance is stable with strong operational resilience.")
    elif health_score >= 60:
        st.info("Enterprise performance is moderate. Strategic monitoring advised.")
    else:
        st.error("Enterprise stability is under pressure. Immediate executive review recommended.")