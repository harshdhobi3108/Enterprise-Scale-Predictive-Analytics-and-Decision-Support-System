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
    # LOAD DATA + GLOBAL CUSTOMER MASTER
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

        # 🔥 GLOBAL CUSTOMER MASTER (FIXED)
        customer_master = (
            data["customers"]
            .sort_values("customer_unique_id")
            .drop_duplicates("customer_unique_id")
            .reset_index(drop=True)
        )

        customer_master["customer_code"] = (
            "CUST-" + (customer_master.index + 1).astype(str).str.zfill(5)
        )

        return segmented_df, segmenter, customer_master

    rfm, segmenter, customer_master = load_rfm()

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
    # EXECUTIVE KPIs
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
    k4.metric("VIP Revenue Contribution", f"{vip_contribution:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================
    # TABS
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

        col1.plotly_chart(fig_dist, use_container_width=True)

        revenue_df = segmenter.revenue_contribution(rfm)

        fig_rev = px.pie(
            revenue_df,
            names="Segment",
            values="Monetary",
            template="plotly_dark"
        )

        col2.plotly_chart(fig_rev, use_container_width=True)

    # ==========================================================
    # TAB 2 — BEHAVIORAL ANALYTICS
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

        st.plotly_chart(fig_3d, use_container_width=True)

    # ==========================================================
    # TAB 3 — CUSTOMER INTELLIGENCE
    # ==========================================================
    with tab3:

        st.subheader("Customer Intelligence Profile")

        # 🔥 MERGE WITH GLOBAL CUSTOMER MASTER
        rfm = rfm.merge(
            customer_master[
                ["customer_unique_id", "customer_code"]
            ],
            on="customer_unique_id",
            how="left"
        )

        rfm_sorted = rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)

        # ✅ CONSISTENT DISPLAY FORMAT
        rfm_sorted["Display_Label"] = (
            rfm_sorted["customer_code"]
            + " | "
            + rfm_sorted["Customer_City"].str.title()
            + " | "
            + rfm_sorted["Customer_State"].str.upper()
        )

        selected_display = st.selectbox(
            "Select Customer",
            rfm_sorted["Display_Label"]
        )

        customer_data = rfm_sorted[
            rfm_sorted["Display_Label"] == selected_display
        ].iloc[0]

        total_revenue = rfm_sorted["Monetary"].sum()

        revenue_percentile = customer_data["Revenue_Percentile"]
        contribution_pct = customer_data["Monetary"] / total_revenue * 100

        segment_avg = (
            rfm_sorted[rfm_sorted["Segment"] == customer_data["Segment"]]["Monetary"].mean()
        )

        pct_vs_segment = (
            (customer_data["Monetary"] - segment_avg) / segment_avg * 100
        )

        # ===============================
        # PRIMARY METRICS
        # ===============================
        c1, c2, c3, c4 = st.columns(4, gap="large")
        c1.metric("Customer Segment", customer_data["Segment"])
        c2.metric("Days Since Last Purchase", int(customer_data["Recency"]))
        c3.metric("Total Orders", int(customer_data["Frequency"]))
        c4.metric("Total Revenue Generated", f"${customer_data['Monetary']:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ===============================
        # INTELLIGENCE METRICS
        # ===============================
        c5, c6, c7, c8 = st.columns(4, gap="large")
        c5.metric("Revenue Percentile", f"{revenue_percentile:.1f}%")
        c6.metric("Revenue Contribution", f"{contribution_pct:.4f}%")
        c7.metric("Segment Avg Revenue", f"${segment_avg:,.2f}")
        c8.metric("Vs Segment Avg", f"{pct_vs_segment:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ===============================
        # STRATEGIC INSIGHT
        # ===============================
        st.markdown(f"""
        <div style="padding:20px;border-radius:15px;
                    background:rgba(59,130,246,0.08);
                    border:1px solid rgba(59,130,246,0.3);">

        <h4 style="margin:0;">Strategic Classification</h4>
        <p style="margin:5px 0;">
        <b>Customer Segment:</b> {customer_data['Segment']} <br>
        <b>Strategic Tier:</b> {customer_data['Strategic_Tier']} <br>
        <b>Recommended Action:</b> {segmenter.recommend_strategy(customer_data['Segment'])}
        </p>

        </div>
        """, unsafe_allow_html=True)