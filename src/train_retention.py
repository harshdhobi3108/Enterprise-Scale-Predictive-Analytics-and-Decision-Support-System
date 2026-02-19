"""
Train Enterprise Retention Model
Customer Lifecycle Intelligence Engine
"""

from data_loader import DataLoader
from retention_features import create_retention_target, build_retention_features
from retention_model import train_retention_model


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
    # STEP 1: Attach customer_unique_id to orders
    # ==========================================================
    orders = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left"
    )

    # ==========================================================
    # STEP 2: Attach customer_unique_id to payments & reviews
    # ==========================================================
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
    # STEP 3: Create Retention Target (Using customer_unique_id)
    # ==========================================================
    print("Creating retention target...")
    retention = create_retention_target(orders)

    # ==========================================================
    # STEP 4: Build Retention Features
    # ==========================================================
    print("Building retention features...")
    features = build_retention_features(orders, payments, reviews)

    # ==========================================================
    # STEP 5: Merge Features + Target
    # ==========================================================
    dataset = features.merge(
        retention,
        on="customer_unique_id"
    )

    dataset = dataset.dropna()

    # ==========================================================
    # STEP 6: Prepare Training Data
    # ==========================================================
    X = dataset.drop(columns=["customer_unique_id", "retained"])
    y = dataset["retained"]

    # Safety check (very important)
    print("\nClass Distribution:")
    print(y.value_counts())

    # ==========================================================
    # STEP 7: Train Model
    # ==========================================================
    print("\nTraining retention model...")
    train_retention_model(X, y)

    print("\nRetention model training complete.")


if __name__ == "__main__":
    main()
