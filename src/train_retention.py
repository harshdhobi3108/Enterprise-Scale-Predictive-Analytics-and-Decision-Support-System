"""
Train Enterprise Retention Model
Customer Lifecycle Intelligence Engine (Production-Safe Version)
"""

import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# ✅ FIXED IMPORTS
from src.data_loader import DataLoader
from src.retention_features import build_retention_features


# ==========================================================
# Create Retention Target (FUTURE WINDOW)
# ==========================================================
def create_retention_target_future(future_orders):

    retention = (
        future_orders
        .groupby("customer_unique_id")
        .size()
        .reset_index(name="future_orders")
    )

    retention["retained"] = 1
    return retention[["customer_unique_id", "retained"]]


# ==========================================================
# Build Pipeline (PRODUCTION SAFE)
# ==========================================================
def build_pipeline(X):

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    return pipeline


# ==========================================================
# Main Training Pipeline
# ==========================================================
def main():

    print("Initializing Data Loader...")
    loader = DataLoader("data/raw")

    print("Loading datasets...")
    data = loader.load_all()

    orders = data["orders"]
    customers = data["customers"]
    payments = data["payments"]
    reviews = data["reviews"]

    # ==========================================================
    # STEP 1: Attach customer_unique_id
    # ==========================================================
    orders = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left"
    )

    payments = payments.merge(
        orders[["order_id", "customer_unique_id"]],
        on="order_id",
        how="left"
    )

    reviews = reviews.merge(
        orders[["order_id", "customer_unique_id"]],
        on="order_id",
        how="left"
    )

    # ==========================================================
    # STEP 2: Convert to datetime
    # ==========================================================
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    # ==========================================================
    # STEP 3: Time-based split (NO LEAKAGE)
    # ==========================================================
    cutoff_date = orders["order_purchase_timestamp"].quantile(0.80)

    print(f"\nCutoff Date: {cutoff_date}")

    past_orders = orders[
        orders["order_purchase_timestamp"] <= cutoff_date
    ]

    future_orders = orders[
        orders["order_purchase_timestamp"] > cutoff_date
    ]

    print(f"Past Orders: {len(past_orders)}")
    print(f"Future Orders: {len(future_orders)}")

    # ==========================================================
    # STEP 4: Feature Engineering
    # ==========================================================
    print("\nBuilding features from past data...")
    features = build_retention_features(
        past_orders,
        payments,
        reviews
    )

    # ==========================================================
    # STEP 5: Target Creation
    # ==========================================================
    print("Creating retention target from future data...")
    retention = create_retention_target_future(future_orders)

    # ==========================================================
    # STEP 6: Merge
    # ==========================================================
    dataset = features.merge(
        retention,
        on="customer_unique_id",
        how="left"
    )

    dataset["retained"] = dataset["retained"].fillna(0)

    # ==========================================================
    # STEP 7: Prepare Data
    # ==========================================================
    X = dataset.drop(columns=["customer_unique_id", "retained"])
    y = dataset["retained"]

    print("\nClass Distribution:")
    print(y.value_counts())

    # ==========================================================
    # STEP 8: Train Pipeline
    # ==========================================================
    print("\nBuilding pipeline...")
    pipeline = build_pipeline(X)

    print("Training model...")
    pipeline.fit(X, y)

    print("\nModel training complete.")

    # ==========================================================
    # STEP 9: SAVE MODEL + FEATURES
    # ==========================================================
    os.makedirs("models", exist_ok=True)

    joblib.dump(pipeline, "models/retention_model.pkl")
    joblib.dump(list(X.columns), "models/model_features.pkl")

    print("Model saved successfully.")

    return pipeline


if __name__ == "__main__":
    main()