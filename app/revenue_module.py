def run_revenue_dashboard():

    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from src.data_loader import DataLoader
    from src.rfm_segmentation import RFMSegmenter

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
    st.markdown("## 💰 Revenue Intelligence")
    st.caption("Customer segmentation and revenue analytics")
    st.markdown("---")

    # ==========================================================
    # KPIs
    # ==========================================================
    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()
    avg_value = rfm["Monetary"].mean()
    vip_revenue = rfm[rfm["Segment"] == "VIP Customers"]["Monetary"].sum()
    vip_contribution = (vip_revenue / total_revenue) * 100

    def card(title, value):
        return f"""
        <div style="background:#111827;padding:18px;border-radius:14px;">
            <div style="color:#9ca3af;font-size:13px">{title}</div>
            <div style="font-size:28px;font-weight:600">{value}</div>
        </div>
        """

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Customers", f"{total_customers:,}"), unsafe_allow_html=True)
    c2.markdown(card("Revenue", f"${total_revenue:,.0f}"), unsafe_allow_html=True)
    c3.markdown(card("Avg Value", f"${avg_value:,.2f}"), unsafe_allow_html=True)
    c4.markdown(card("VIP Contribution", f"{vip_contribution:.2f}%"), unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================================
    # SEGMENT DISTRIBUTION (BAR)
    # ==========================================================
    st.markdown("### 📊 Customer Segmentation")

    segment_counts = rfm["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    fig = px.bar(segment_counts, x="Segment", y="Count", color="Segment")
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # REVENUE CONTRIBUTION (BAR)
    # ==========================================================
    st.markdown("### 💰 Revenue Contribution")

    revenue_df = segmenter.revenue_contribution(rfm)

    fig2 = px.bar(revenue_df, x="Segment", y="Monetary", color="Segment")
    st.plotly_chart(fig2, use_container_width=True)