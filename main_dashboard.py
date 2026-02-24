"""
Enterprise Predictive Analytics Suite
Production-Grade AI Intelligence Platform
"""

import streamlit as st
from datetime import datetime
from streamlit_option_menu import option_menu

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Enterprise Analytics Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# GLOBAL ENTERPRISE STYLING
# ==========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0b1220 0%, #0e1627 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    display: none;
}

h1, h2, h3 {
    font-weight: 600 !important;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 22px;
    border-radius: 14px;
}

.stButton button {
    background-color: #1f2937;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    font-weight: 500;
}

.stButton button:hover {
    background-color: #111827;
}

.nav-container {
    padding: 8px;
    border-radius: 12px;
    background-color: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 25px;
}

.footer {
    text-align:center;
    font-size:13px;
    color:#6b7280;
    padding:15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# MODULE IMPORTS
# ==========================================================

try:
    from app.command_center_module import run_command_center
    from app.delivery_module import run_delivery_dashboard
    from app.revenue_module import run_revenue_dashboard
    from app.retention_module import run_retention_dashboard
    from app.ai_assistant.assistant_ui import run_ai_assistant   # ✅ ADD THIS
except Exception as e:
    st.error(f"Module import failed: {e}")
    st.stop()

# ==========================================================
# HEADER
# ==========================================================

st.markdown(f"""
<div style="
    padding:16px 24px;
    border-radius:14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom:30px;
    display:flex;
    justify-content:space-between;
    align-items:center;
">
    <div>
        <div style="font-size:20px;font-weight:600;">
            Enterprise Predictive Analytics Suite
        </div>
        <div style="font-size:13px;color:#94a3b8;">
            AI-Driven Decision Intelligence Platform
        </div>
    </div>
    <div style="font-size:13px;color:#94a3b8;">
        {datetime.now().strftime('%d %B %Y | %H:%M')}
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# NAVIGATION
# ==========================================================

selected_page = option_menu(
    menu_title=None,
    options=[
        "Command Center",
        "Delivery Risk",
        "Revenue Intelligence",
        "Lifecycle Intelligence"
    ],
    icons=["speedometer2", "truck", "currency-dollar", "people"],
    orientation="horizontal"
)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# ROUTING
# ==========================================================

if selected_page == "Command Center":
    run_command_center()

elif selected_page == "Delivery Risk":
    run_delivery_dashboard()

elif selected_page == "Revenue Intelligence":
    run_revenue_dashboard()

elif selected_page == "Lifecycle Intelligence":
    run_retention_dashboard()

# ==========================================================
# FLOATING AI ASSISTANT (ALWAYS VISIBLE)
# ==========================================================

run_ai_assistant()   # ✅ THIS MAKES IT APPEAR

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    © 2026 Enterprise Analytics Platform | Internal Intelligence System
</div>
""", unsafe_allow_html=True)