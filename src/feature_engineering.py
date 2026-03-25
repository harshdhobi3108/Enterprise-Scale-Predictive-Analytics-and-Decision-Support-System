"""
Enterprise Feature Engineering Module
Delivery Delay Prediction System
"""

import pandas as pd
import logging
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class DeliveryFeatureEngineer(BaseEstimator, TransformerMixin):

    FEATURE_COLUMNS: List[str] = [
        "purchase_hour",
        "purchase_dayofweek",
        "purchase_month",
        "approval_delay_hours",
        "carrier_delay_hours",
        "estimated_delivery_days",
        "total_payment_value",
        "payment_installments",
        "customer_state"
    ]

    TARGET_COLUMN = "is_delayed"

    DATETIME_COLUMNS = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    def __init__(
        self,
        predict_before_delivery: bool = True,
        max_approval_hours: int = 168,
        max_carrier_hours: int = 720,
        max_delivery_days: int = 60
    ) -> None:
        self.predict_before_delivery = predict_before_delivery
        self.max_approval_hours = max_approval_hours
        self.max_carrier_hours = max_carrier_hours
        self.max_delivery_days = max_delivery_days
        self.df: Optional[pd.DataFrame] = None

    # ==========================================================
    # CORE METHODS
    # ==========================================================

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.df = X.copy()

        self._parse_datetimes()
        self._create_target()
        self._create_time_features()
        self._clean_data()

        return self.df   # ✅ FIXED (no final dataset here)

    # ==========================================================
    # INTERNAL METHODS
    # ==========================================================

    def _parse_datetimes(self):
        for col in self.DATETIME_COLUMNS:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

    def _create_target(self):
        self.df = self.df.dropna(
            subset=[
                "order_delivered_customer_date",
                "order_estimated_delivery_date"
            ]
        )

        self.df[self.TARGET_COLUMN] = (
            self.df["order_delivered_customer_date"]
            > self.df["order_estimated_delivery_date"]
        ).astype(int)

    def _create_time_features(self):
        purchase = self.df["order_purchase_timestamp"]

        self.df["purchase_hour"] = purchase.dt.hour
        self.df["purchase_dayofweek"] = purchase.dt.dayofweek
        self.df["purchase_month"] = purchase.dt.month

        self.df["approval_delay_hours"] = (
            (self.df["order_approved_at"] - purchase)
            .dt.total_seconds() / 3600
        )

        self.df["carrier_delay_hours"] = (
            (self.df["order_delivered_carrier_date"]
             - self.df["order_approved_at"])
            .dt.total_seconds() / 3600
        )

        self.df["estimated_delivery_days"] = (
            (self.df["order_estimated_delivery_date"] - purchase)
            .dt.total_seconds() / (3600 * 24)
        )

    def _clean_data(self):
        self.df = self.df[
            self.df["approval_delay_hours"].between(0, self.max_approval_hours)
            & self.df["carrier_delay_hours"].between(0, self.max_carrier_hours)
            & self.df["estimated_delivery_days"].between(0, self.max_delivery_days)
        ]

    def merge_additional_data(self, payments_df, customers_df):

        payment_agg = (
            payments_df.groupby("order_id")
            .agg(
                total_payment_value=("payment_value", "sum"),
                payment_installments=("payment_installments", "max")
            )
            .reset_index()
        )

        self.df = self.df.merge(payment_agg, on="order_id", how="left")

        self.df = self.df.merge(
            customers_df[["customer_id", "customer_state"]],
            on="customer_id",
            how="left"
        )

        self.df["total_payment_value"] = self.df["total_payment_value"].fillna(0)
        self.df["payment_installments"] = self.df["payment_installments"].fillna(1)
        self.df["customer_state"] = self.df["customer_state"].fillna("Unknown")

    def get_final_dataset(self):

        final_columns = self.FEATURE_COLUMNS.copy()

        if self.TARGET_COLUMN in self.df.columns:
            final_columns.append(self.TARGET_COLUMN)

        return self.df[final_columns]