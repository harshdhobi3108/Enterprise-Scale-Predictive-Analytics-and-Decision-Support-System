import streamlit as st
from app.ai_assistant.intent_router import detect_intent
import random


def generate_dynamic_insight(intent: str):

    revenue_growth = round(random.uniform(-4, 7), 2)
    churn_rate = round(random.uniform(6, 18), 2)
    delivery_risk = random.randint(12, 48)

    if intent == "greeting":
        return "AI Intelligence Copilot active. Specify a business domain."

    if intent == "revenue_analysis":
        direction = "increased" if revenue_growth > 0 else "declined"
        return f"Revenue {direction} by {abs(revenue_growth)}% this quarter."

    if intent == "churn_analysis":
        return f"Churn rate currently at {churn_rate}%."

    if intent == "delivery_analysis":
        return f"Delivery risk exposure at {delivery_risk}%."

    if intent == "segmentation_analysis":
        return "Enterprise segment delivering highest profitability."

    return "Supported domains: Revenue, Churn, Delivery, Segmentation."


def run_ai_assistant():

    if "copilot_open" not in st.session_state:
        st.session_state.copilot_open = False

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -----------------------
    # CSS
    # -----------------------
    st.markdown("""
    <style>

    /* Floating Button */
    div[data-testid="stButton"] > button:first-child {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563eb, #1e3a8a);
        color: white;
        font-weight: 600;
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
        z-index: 1002;
    }

    /* Panel */
    .copilot-panel {
        position: fixed;
        top: 0;
        right: 0;
        width: 420px;
        height: 100vh;
        background: #0f172a;
        border-left: 1px solid rgba(255,255,255,0.08);
        box-shadow: -15px 0 40px rgba(0,0,0,0.6);
        padding: 20px;
        display: flex;
        flex-direction: column;
        z-index: 1001;
    }

    .chat-area {
        flex: 1;
        overflow-y: auto;
        margin-bottom: 15px;
    }

    .input-area {
        border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

    # -----------------------
    # Button
    # -----------------------
    if st.button("AI", key="copilot_toggle"):
        st.session_state.copilot_open = not st.session_state.copilot_open

    # -----------------------
    # Panel
    # -----------------------
    if st.session_state.copilot_open:

        st.markdown('<div class="copilot-panel">', unsafe_allow_html=True)

        st.markdown("### AI Intelligence Copilot")
        st.caption("Conversational Business Intelligence")

        # Chat Area
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)

        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**AI:** {message}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Input Area
        st.markdown('<div class="input-area">', unsafe_allow_html=True)

        user_input = st.text_input(
            "Message",
            key="custom_input",
            label_visibility="collapsed",
            placeholder="Ask about revenue, churn, delivery..."
        )

        if user_input:
            intent = detect_intent(user_input)
            response = generate_dynamic_insight(intent)

            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response))

            st.session_state.custom_input = ""
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)