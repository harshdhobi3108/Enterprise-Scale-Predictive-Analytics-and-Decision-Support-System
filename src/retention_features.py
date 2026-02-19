"""
Enterprise Retention Feature Engineering
Builds customer-level features for retention prediction
"""

import pandas as pd


def create_retention_target(
    orders: pd.DataFrame,
    window_days: int = 120
) -> pd.DataFrame:
    """
    1 -> Customer purchased again within window_days
    0 -> Otherwise
    """

    orders = orders.sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    )

    orders["next_purchase"] = (
        orders.groupby("customer_unique_id")["order_purchase_timestamp"]
        .shift(-1)
    )

    orders["days_to_next"] = (
        orders["next_purchase"] - orders["order_purchase_timestamp"]
    ).dt.days

    retention = (
        orders.groupby("customer_unique_id")["days_to_next"]
        .apply(lambda x: int(((x > 0) & (x <= window_days)).any()))
        .reset_index()
    )

    retention.columns = ["customer_unique_id", "retained"]

    return retention


def build_retention_features(
    orders: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame
) -> pd.DataFrame:
    """
    Build numeric lifecycle features only (model-safe).
    """

    # ======================================================
    # RFM FEATURES
    # ======================================================
    rfm = orders.groupby("customer_unique_id").agg({
        "order_id": "count",
        "order_purchase_timestamp": ["min", "max"]
    })

    rfm.columns = [
        "frequency",
        "first_purchase",
        "last_purchase"
    ]

    # Recency in days
    rfm["recency_days"] = (
        orders["order_purchase_timestamp"].max()
        - rfm["last_purchase"]
    ).dt.days

    # Customer lifetime in days
    rfm["customer_lifetime_days"] = (
        rfm["last_purchase"] - rfm["first_purchase"]
    ).dt.days

    # ======================================================
    # PAYMENT FEATURES
    # ======================================================
    payment_agg = payments.groupby(
        "customer_unique_id"
    ).agg({
        "payment_value": ["sum", "mean"]
    })

    payment_agg.columns = [
        "total_spent",
        "avg_order_value"
    ]

    # ======================================================
    # REVIEW FEATURES
    # ======================================================
    review_agg = reviews.groupby(
        "customer_unique_id"
    )["review_score"].mean()

    review_agg = review_agg.to_frame("avg_review_score")

    # ======================================================
    # MERGE ALL
    # ======================================================
    features = (
        rfm
        .join(payment_agg)
        .join(review_agg)
    )

    # ------------------------------------------------------
    # DROP RAW DATETIME COLUMNS (CRITICAL FIX)
    # ------------------------------------------------------
    features = features.drop(
        columns=["first_purchase", "last_purchase"]
    )

    features = features.fillna(0)

    features = features.reset_index()

    return features
