def run_retention_dashboard():

    import os
    import joblib
    import streamlit as st

    st.title("Customer Lifecycle Intelligence")

    # ==========================================================
    # PATH SETUP
    # ==========================================================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_PATH = os.path.join(BASE_DIR, "models", "delivery_model.pkl")
    FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")
    THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.pkl")

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    # ==========================================================
    # LOAD FEATURES + THRESHOLD
    # ==========================================================
    try:
        model_features = joblib.load(FEATURES_PATH)
    except:
        st.error("model_features.pkl missing")
        st.stop()

    if os.path.exists(THRESHOLD_PATH):
        threshold = joblib.load(THRESHOLD_PATH)
    else:
        threshold = 0.5

    # ==========================================================
    # LOAD DATA
    # ==========================================================
    features_df, customer_info = load_retention_features()

    features_df = features_df.merge(
        customer_info[
            ["customer_unique_id", "customer_code", "customer_city", "customer_state"]
        ],
        on="customer_unique_id",
        how="left"
    )

    features_df["customer_display"] = (
        features_df["customer_code"] +
        " | " +
        features_df["customer_city"].str.title() +
        " | " +
        features_df["customer_state"]
    )

    # ==========================================================
    # 🔥 FEATURE ALIGNMENT (CRITICAL FIX)
    # ==========================================================
    X_all = features_df.drop(columns=[
        "customer_unique_id",
        "customer_code",
        "customer_city",
        "customer_state",
        "customer_display"
    ], errors="ignore")

    # Add missing columns
    for col in model_features:
        if col not in X_all.columns:
            X_all[col] = 0

    # Ensure correct order
    X_all = X_all[model_features]

    # ==========================================================
    # PREDICTIONS
    # ==========================================================
    try:
        probabilities = model.predict_proba(X_all)[:, 1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    features_df["retention_probability"] = probabilities

    # ==========================================================
    # TOP CUSTOMERS
    # ==========================================================
    st.subheader("Top Likely To Retain Customers")

    top_customers = (
        features_df
        .sort_values("retention_probability", ascending=False)
        .drop_duplicates(subset=["customer_code"])
        .head(5)
    )

    st.dataframe(
        top_customers[
            ["customer_display", "retention_probability"]
        ].rename(columns={
            "customer_display": "Customer",
            "retention_probability": "Retention Probability"
        }),
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # CUSTOMER SELECT
    # ==========================================================
    customer_map = dict(
        zip(features_df["customer_display"], features_df["customer_unique_id"])
    )

    selected_display = st.selectbox(
        "Select Customer",
        list(customer_map.keys())
    )

    selected_customer = customer_map[selected_display]

    # ==========================================================
    # CUSTOMER DATA
    # ==========================================================
    customer_data = features_df[
        features_df["customer_unique_id"] == selected_customer
    ].drop(columns=[
        "customer_unique_id",
        "customer_code",
        "customer_city",
        "customer_state",
        "customer_display",
        "retention_probability"
    ], errors="ignore")

    # 🔥 ALIGN FEATURES AGAIN
    for col in model_features:
        if col not in customer_data.columns:
            customer_data[col] = 0

    customer_data = customer_data[model_features]

    # ==========================================================
    # SINGLE PREDICTION
    # ==========================================================
    probability = model.predict_proba(customer_data)[0][1]

    st.metric("Retention Probability (%)", f"{probability * 100:.2f}")

    # ==========================================================
    # BUSINESS LOGIC
    # ==========================================================
    if probability < threshold:
        st.error("High Churn Risk")
    else:
        st.success("Likely to Retain")

    # ==========================================================
    # SHAP (OPTIONAL SAFE)
    # ==========================================================
    st.subheader("Why this prediction?")

    try:
        explainer = RetentionExplainer(MODEL_PATH)
        shap_values, X_named = explainer.explain_instance(customer_data)

        if shap_values is not None:
            import matplotlib.pyplot as plt
            import shap

            fig, ax = plt.subplots()

            shap.plots._waterfall.waterfall_legacy(
                explainer.explainer.expected_value[1],
                shap_values[1][0],
                X_named.iloc[0],
                show=False
            )

            st.pyplot(fig)
        else:
            st.info("SHAP not available")

    except Exception as e:
        st.warning(f"SHAP error: {e}")