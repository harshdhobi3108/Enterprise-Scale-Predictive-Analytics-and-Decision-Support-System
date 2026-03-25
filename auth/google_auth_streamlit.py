import streamlit as st
import requests
from streamlit_oauth import OAuth2Component
import os
from dotenv import load_dotenv

# ==========================================================
# LOAD ENV VARIABLES
# ==========================================================
load_dotenv()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USER_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8501")

# Default fallback avatar
DEFAULT_AVATAR = "https://cdn-icons-png.flaticon.com/512/149/149071.png"

# ==========================================================
# OAUTH COMPONENT
# ==========================================================
oauth2 = OAuth2Component(
    CLIENT_ID,
    CLIENT_SECRET,
    AUTHORIZE_URL,
    TOKEN_URL
)

# ==========================================================
# GOOGLE LOGIN FUNCTION
# ==========================================================
def google_login():
    result = oauth2.authorize_button(
        name="Login with Google",
        redirect_uri=REDIRECT_URI,
        scope="openid email profile",
        key="google_login",
        use_container_width=True
    )

    if result and "token" in result:
        token = result["token"]["access_token"]

        response = requests.get(
            USER_INFO_URL,
            headers={"Authorization": f"Bearer {token}"}
        )

        if response.status_code != 200:
            st.error("Failed to fetch user info from Google.")
            return False

        user_info = response.json()

        # Store user info safely
        st.session_state["user_email"] = user_info.get("email", "")
        st.session_state["user_name"] = user_info.get("name", "User")

        # Avatar handling
        picture = user_info.get("picture", DEFAULT_AVATAR)

        # Improve Google avatar resolution
        if "googleusercontent" in picture:
            picture = picture.replace("=s96", "=s200")

        st.session_state["user_picture"] = picture

        return True

    return False