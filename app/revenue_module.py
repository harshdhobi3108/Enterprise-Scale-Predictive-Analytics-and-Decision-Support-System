def run_revenue_dashboard():

    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import numpy as np
    from src.data_loader import DataLoader
    from src.rfm_segmentation import RFMSegmenter

    # ==========================================================
    # PREMIUM ENTERPRISE STYLING
    # ==========================================================
    st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(135deg, #0b1220, #111827);
    }

    h1 {
        font-size: 38px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }

    h2, h3 {
        color: #e2e8f0 !important;
        margin-top: 10px;
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 28px;
        border-radius: 18px;
        backdrop-filter: blur(6px);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
    }

    </style>
    """, unsafe_allow_html=True)

    # ==========================================================
    # LOAD DATA
    # ==========================================================
    @st.cache_data
    def load_rfm():
        loader = DataLoader(data_dir="data/raw")
        data = loader.load_all()

        segmenter = RFMSegmenter(
            data["orders"],
            data["payments"],
            data["customers"]
        )

        rfm_df = segmenter.build_rfm()
        segmented_df = segmenter.segment(rfm_df, n_clusters=4)

        return segmented_df, segmenter

    rfm, segmenter = load_rfm()

    # ==========================================================
    # HEADER
    # ==========================================================
    st.markdown("""
    <div style="padding: 20px 0 5px 0;">
        <h1>Enterprise Customer Revenue Intelligence</h1>
        <p style="color:#94a3b8;font-size:16px;">
            Segmentation • Revenue Concentration • Behavioral Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    # ==========================================================
    # EXECUTIVE KPI SECTION
    # ==========================================================
    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()
    avg_value = rfm["Monetary"].mean()
    vip_revenue = rfm[rfm["Segment"] == "VIP Customers"]["Monetary"].sum()
    vip_contribution = (vip_revenue / total_revenue) * 100

    k1, k2, k3, k4 = st.columns(4, gap="large")
    k1.metric("Total Customers", f"{total_customers:,}")
    k2.metric("Total Revenue", f"${total_revenue:,.0f}")
    k3.metric("Avg Customer Value", f"${avg_value:,.2f}")
    k4.metric("VIP Contribution", f"{vip_contribution:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # TABBED EXECUTIVE LAYOUT
    # ==========================================================
    tab1, tab2, tab3 = st.tabs([
        "📊 Segmentation Overview",
        "📈 Behavioral Analytics",
        "👤 Customer Intelligence"
    ])

    # ==========================================================
    # TAB 1 — SEGMENTATION
    # ==========================================================
    with tab1:

        col1, col2 = st.columns(2, gap="large")

        segment_counts = rfm["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]

        fig_dist = px.bar(
            segment_counts,
            x="Segment",
            y="Count",
            color="Segment",
            template="plotly_dark"
        )

        fig_dist.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font=dict(color="#e2e8f0")
        )

        col1.plotly_chart(fig_dist, width="stretch")

        revenue_df = segmenter.revenue_contribution(rfm)

        fig_rev = px.pie(
            revenue_df,
            names="Segment",
            values="Monetary",
            template="plotly_dark"
        )

        fig_rev.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font=dict(color="#e2e8f0")
        )

        col2.plotly_chart(fig_rev, width="stretch")

    # ==========================================================
    # TAB 2 — 3D ANALYTICS
    # ==========================================================
    with tab2:

        st.subheader("3D Behavioral Segmentation Map")

        fig_3d = px.scatter_3d(
            rfm,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Segment",
            template="plotly_dark"
        )

        fig_3d.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font=dict(color="#e2e8f0")
        )

        st.plotly_chart(fig_3d, width="stretch")

    # ==========================================================
    # TAB 3 — CUSTOMER PROFILE PANEL
    # ==========================================================
    with tab3:

        st.subheader("Customer Intelligence Profile")

        selected_customer = st.selectbox(
            "Select Customer",
            rfm["customer_unique_id"].sample(300, random_state=42)
        )

        customer_data = rfm[rfm["customer_unique_id"] == selected_customer].iloc[0]

        total_revenue = rfm["Monetary"].sum()

        revenue_percentile = (
            (rfm["Monetary"] < customer_data["Monetary"]).mean() * 100
        )

        contribution_pct = (
            customer_data["Monetary"] / total_revenue * 100
        )

        c1, c2, c3, c4 = st.columns(4, gap="large")
        c1.metric("Segment", customer_data["Segment"])
        c2.metric("Recency (Days)", int(customer_data["Recency"]))
        c3.metric("Frequency", int(customer_data["Frequency"]))
        c4.metric("Monetary Value", f"${customer_data['Monetary']:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        c5, c6 = st.columns(2, gap="large")
        c5.metric("Revenue Percentile", f"{revenue_percentile:.1f}%")
        c6.metric("Revenue Contribution", f"{contribution_pct:.4f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        st.info(
            f"This customer belongs to **{customer_data['Segment']}**. "
            f"They rank in the top {100 - revenue_percentile:.1f}% of customers by revenue. "
            f"Strategic engagement should align with their behavioral classification."
        )
