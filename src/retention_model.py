"""
Enterprise Retention Model Training
Handles extreme class imbalance properly
"""

import joblib
import os
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    average_precision_score
)
from xgboost import XGBClassifier


def train_retention_model(
    X,
    y,
    model_path="models/retention_model.pkl"
):

    # ==========================================================
    # Train-Test Split
    # ==========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ==========================================================
    # Handle Class Imbalance
    # ==========================================================
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]

    print("\nTraining Class Distribution:")
    print(counter)
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    # ==========================================================
    # Preprocessing
    # ==========================================================
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features)
    ])

    # ==========================================================
    # XGBoost Model (Imbalance-Aware)
    # ==========================================================
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # ==========================================================
    # Train
    # ==========================================================
    pipeline.fit(X_train, y_train)

    # ==========================================================
    # Evaluation
    # ==========================================================
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print("\nModel Performance:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ==========================================================
    # Save Model
    # ==========================================================
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"\nModel saved at: {model_path}")

    return pipeline
