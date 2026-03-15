"""
Enterprise Predictive Analytics System
Streamlit Authentication + Dashboard Launcher
"""

import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Enterprise Predictive Analytics Suite",
    layout="wide"
)


# =====================================================
# LOGIN STATE
# =====================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# =====================================================
# LOGIN PAGE
# =====================================================

if not st.session_state.authenticated:

    st.markdown("""
    <div style='text-align:center;margin-top:120px'>
        <h1>Enterprise Predictive Analytics Suite</h1>
        <p>Secure access required</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):

        if username == "admin" and password == "admin123":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()


# =====================================================
# DASHBOARD LAUNCH
# =====================================================

st.success("Login successful")

st.write("Open the analytics dashboard below.")

st.markdown(
    """
    ### Enterprise Analytics Dashboard

    Click below to open the full dashboard.
    """
)

st.markdown(
    "[Open Dashboard](http://localhost:8501)"
)