import pandas as pd
import joblib
import os
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV


def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(BASE_DIR, "data", "processed", "delivery_features.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at: {data_path}")

    return pd.read_csv(data_path)


def prepare_features(df):

    feature_cols = [
        "purchase_hour",
        "purchase_dayofweek",
        "purchase_month",
        "total_payment_value",
        "payment_installments"
    ]

    df["is_delayed"] = (
        (df["carrier_delay_hours"] > 20) |
        (df["approval_delay_hours"] > 8) |
        (df["estimated_delivery_days"] > 9)
    ).astype(int)

    X = df[feature_cols]
    y = df["is_delayed"]

    return X, y, feature_cols


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )

    model = CalibratedClassifierCV(base_model, method="sigmoid")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n📊 MODEL PERFORMANCE")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, probs):.4f}")

    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_score = 0

    for t in thresholds:
        temp_preds = (probs >= t).astype(int)
        score = f1_score(y_test, temp_preds)

        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"\n🔥 Best Threshold: {best_threshold:.2f}")

    return model, best_threshold


def save_model(model, feature_list, threshold):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/delivery_model.pkl")
    joblib.dump(feature_list, "models/model_features.pkl")
    joblib.dump(threshold, "models/threshold.pkl")

    print("\n✅ Model, features & threshold saved successfully!")


def main():
    print("🚀 Training Production Delivery Model...")

    df = load_data()
    X, y, feature_list = prepare_features(df)

    model, threshold = train_model(X, y)

    save_model(model, feature_list, threshold)

    print("\n🎯 Training completed successfully!")


if __name__ == "__main__":
    main()