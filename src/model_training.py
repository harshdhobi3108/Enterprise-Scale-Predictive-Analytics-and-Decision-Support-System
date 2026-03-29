import pandas as pd
import joblib
import os
from lightgbm import LGBMClassifier


def load_data():
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(BASE_DIR, "data", "processed", "delivery_features.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    return df


# Feature preparation
def prepare_features(df):
    feature_cols = [
        "purchase_hour",
        "purchase_dayofweek",
        "purchase_month",
        "approval_delay_hours",
        "carrier_delay_hours",
        "estimated_delivery_days",
        "total_payment_value",
        "payment_installments"
    ]

    target_col = "is_delayed"  # UPDATE if needed

    X = df[feature_cols]
    y = df[target_col]

    return X, y, feature_cols


# Train model
def train_model(X, y):
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)
    return model


# Save model bundle
def save_model(model, feature_list):
    os.makedirs("models", exist_ok=True)

    bundle = {
        "model": model,
        "features": feature_list
    }

    joblib.dump(bundle, "models/model_features.pkl")
    print("✅ Model saved successfully!")


# Main execution
def main():
    print("🚀 Training started...")

    df = load_data()
    X, y, feature_list = prepare_features(df)

    model = train_model(X, y)

    save_model(model, feature_list)

    print("🎯 Training completed!")


if __name__ == "__main__":
    main()