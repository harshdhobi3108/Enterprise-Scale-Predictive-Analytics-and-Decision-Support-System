"""
Train Enterprise Retention Model
Customer Lifecycle Intelligence Engine (Leakage-Free Version)
"""

import pandas as pd
from data_loader import DataLoader
from retention_features import build_retention_features
from retention_model import train_retention_model


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
    # STEP 3: Time-based split (CRITICAL FIX)
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
    # STEP 4: Build Features ONLY from past
    # ==========================================================
    print("\nBuilding features from past data...")
    features = build_retention_features(
        past_orders,
        payments,
        reviews
    )

    # ==========================================================
    # STEP 5: Create Target from future
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

    # Customers with no future orders = churn (0)
    dataset["retained"] = dataset["retained"].fillna(0)

    # ==========================================================
    # STEP 7: Prepare Training Data
    # ==========================================================
    X = dataset.drop(columns=["customer_unique_id", "retained"])
    y = dataset["retained"]

    print("\nClass Distribution:")
    print(y.value_counts())

    # ==========================================================
    # STEP 8: Train Model
    # ==========================================================
    print("\nTraining retention model...")
    model = train_retention_model(X, y)

    print("\nRetention model training complete.")

    return model


if __name__ == "__main__":
    main()