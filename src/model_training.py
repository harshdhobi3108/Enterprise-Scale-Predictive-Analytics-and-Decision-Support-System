import pandas as pd
import joblib
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================================
# LOAD DATA
# ==========================================================
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(BASE_DIR, "data", "processed", "delivery_features.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    return df


# ==========================================================
# FEATURE PREPARATION
# ==========================================================
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

    # ✅ Ensure target exists
    if "is_delayed" not in df.columns:
        df["is_delayed"] = (df["carrier_delay_hours"] > 24).astype(int)

    X = df[feature_cols]
    y = df["is_delayed"]

    return X, y, feature_cols


# ==========================================================
# TRAIN MODEL
# ==========================================================
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ✅ Evaluate (basic)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"✅ Model Accuracy: {acc:.4f}")

    return model


# ==========================================================
# SAVE MODEL (FIXED 🔥)
# ==========================================================
def save_model(model, feature_list):

    os.makedirs("models", exist_ok=True)

    # ✅ Save model separately
    joblib.dump(model, "models/delivery_model.pkl")

    # ✅ Save features separately
    joblib.dump(feature_list, "models/model_features.pkl")

    print("✅ Model & features saved successfully!")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("🚀 Training Delivery Model...")

    df = load_data()
    X, y, feature_list = prepare_features(df)

    model = train_model(X, y)

    save_model(model, feature_list)

    print("🎯 Training completed successfully!")


if __name__ == "__main__":
    main()