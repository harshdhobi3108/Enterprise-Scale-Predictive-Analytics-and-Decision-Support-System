"""
Enterprise Data Processing Pipeline
"""

from src.data_loader import DataLoader
from src.feature_engineering import DeliveryFeatureEngineer
import os


def run_data_pipeline(data_path="data/raw", save_path="data/processed"):

    print("🚀 Running Data Pipeline...")

    # Create processed folder if not exists
    os.makedirs(save_path, exist_ok=True)

    # ==========================================================
    # LOAD DATA
    # ==========================================================
    loader = DataLoader(data_path)
    datasets = loader.load_all()

    orders = datasets["orders"]
    payments = datasets["payments"]
    customers = datasets["customers"]

    # ==========================================================
    # FEATURE ENGINEERING PIPELINE
    # ==========================================================
    fe = DeliveryFeatureEngineer()

    # Step 1: Base features
    fe.transform(orders)

    # Step 2: Merge extra data
    fe.merge_additional_data(payments, customers)

    # Step 3: Final dataset
    df = fe.get_final_dataset()

    # ==========================================================
    # SAVE OUTPUT
    # ==========================================================
    output_path = f"{save_path}/delivery_features.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Processed data saved at: {output_path}")

    return df


if __name__ == "__main__":
    run_data_pipeline()