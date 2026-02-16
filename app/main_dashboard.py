"""
Enterprise Predictive Analytics Suite
Operational Risk + Revenue Intelligence
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from app.delivery_module import run_delivery_dashboard
from app.revenue_module import run_revenue_dashboard

# ==========================================================
# Page Configuration (ONLY HERE)
# ==========================================================
st.set_page_config(
    page_title="Enterprise Analytics Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# Sidebar Navigation
# ==========================================================
st.sidebar.title("Enterprise Analytics Suite")

page = st.sidebar.radio(
    "Select Intelligence Module",
    [
        "Delivery Risk Intelligence",
        "Customer Revenue Intelligence"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Production-Grade Predictive Systems")

# ==========================================================
# Route Modules
# ==========================================================
if page == "Delivery Risk Intelligence":
    run_delivery_dashboard()

elif page == "Customer Revenue Intelligence":
    run_revenue_dashboard()
