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

    import numpy as np

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

    print("\nTraining Class Distribution:", counter)

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
    # XGBoost (with Early Stopping)
    # ==========================================================
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
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
    # Fit with Early Stopping
    # ==========================================================
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_test_transformed, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )

    # Reattach trained components
    pipeline.named_steps["preprocessor"] = preprocessor
    pipeline.named_steps["classifier"] = model

    # ==========================================================
    # Evaluation
    # ==========================================================
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print("\nModel Performance:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    # ==========================================================
    # Optimal Threshold (IMPORTANT)
    # ==========================================================
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_score = 0

    for t in thresholds:
        preds = (y_pred_proba > t).astype(int)
        score = average_precision_score(y_test, preds)

        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"\nOptimal Threshold: {best_threshold:.2f}")

    y_pred = (y_pred_proba > best_threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ==========================================================
    # Save Everything (Model + Threshold)
    # ==========================================================
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump({
        "model": pipeline,
        "threshold": best_threshold
    }, model_path)

    print(f"\nModel saved at: {model_path}")

    return pipeline
