"""
Enterprise Predictive Analytics Suite
Production-Grade AI Intelligence Platform
"""
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from auth.google_auth_streamlit import google_login

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Enterprise Analytics Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# IST TIME FIX (GLOBAL)
# ==========================================================

current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
formatted_time = current_time.strftime('%d %B %Y | %H:%M')

# ==========================================================
# LOGIN SYSTEM
# ==========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:

    st.title("Enterprise Predictive Analytics Suite")
    st.caption("Secure Intelligence Platform")

    st.markdown("### Sign in with your Google Account")

    if google_login():
        st.session_state.authenticated = True
        st.rerun()

    st.stop()

# ==========================================================
# GLOBAL STYLING
# ==========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg, #0b1220 0%, #0e1627 100%);
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    display: none;
}

.dashboard-header {
    padding: 24px 32px;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 28px;
    display:flex;
    justify-content:space-between;
    align-items:center;
}

.dashboard-title {
    font-size:26px;
    font-weight:700;
}

.dashboard-subtitle {
    font-size:14px;
    color:#9ca3af;
}

.user-info {
    text-align:right;
    font-size:14px;
    color:#9ca3af;
}

.user-avatar {
    width:44px;
    height:44px;
    border-radius:50%;
}

.nav-container {
    padding:16px 20px;
    border-radius:14px;
    background: rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:40px;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 22px;
    border-radius: 14px;
}

.stButton button {
    background-color:#1f2937;
    border:1px solid rgba(255,255,255,0.1);
    border-radius:8px;
    font-weight:500;
}

.stButton button:hover {
    background-color:#111827;
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
# LOGOUT BUTTON
# ==========================================================

col1, col2 = st.columns([10,1])

with col2:
    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==========================================================
# MODULE IMPORTS
# ==========================================================

from app.command_center_module import run_command_center
from app.delivery_module import run_delivery_dashboard
from app.revenue_module import run_revenue_dashboard
from app.retention_module import run_retention_dashboard

# ==========================================================
# USER SESSION DATA
# ==========================================================

user_name = st.session_state.get("user_name", "User")
user_picture = st.session_state.get(
    "user_picture",
    "https://cdn-icons-png.flaticon.com/512/149/149071.png"
)

# ==========================================================
# HEADER (FIXED TIME)
# ==========================================================

st.markdown(f"""
<div style="
    padding:24px 32px;
    border-radius:16px;
    background: rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:28px;
    display:flex;
    justify-content:space-between;
    align-items:center;
">

<div>
<div style="font-size:26px;font-weight:700;letter-spacing:0.4px;">
Enterprise Predictive Analytics Suite
</div>

<div style="font-size:14px;color:#9ca3af;margin-top:4px;">
AI-Driven Decision Intelligence Platform
</div>
</div>

<div style="display:flex;align-items:center;gap:12px">

<div style="text-align:right;font-size:14px;color:#9ca3af;">
Logged in as <b>{user_name}</b><br>
{formatted_time}
</div>

<img src="{user_picture}" 
style="width:44px;height:44px;border-radius:50%;border:1px solid rgba(255,255,255,0.2);">

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
    icons=["speedometer2","truck","currency-dollar","people"],
    orientation="horizontal"
)

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
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""
<div class="footer">
© 2026 Enterprise Analytics Platform | Internal Intelligence System
</div>
""", unsafe_allow_html=True)