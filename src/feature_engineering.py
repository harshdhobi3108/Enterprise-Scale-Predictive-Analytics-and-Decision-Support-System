"""
Enterprise Feature Engineering Module
Delivery Delay Prediction System

Production-Ready Version
"""

import pandas as pd
import logging
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class DeliveryFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Enterprise-grade feature engineering pipeline.

    Features:
    - Safe datetime parsing
    - Leakage-aware feature handling
    - Outlier filtering
    - Missing value imputation
    - Feature registry enforcement
    - Sklearn pipeline compatible
    """

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
        """
        Args:
            predict_before_delivery: Avoid leakage features if True
            max_approval_hours: Upper bound for approval delay
            max_carrier_hours: Upper bound for carrier delay
            max_delivery_days: Upper bound for delivery days
        """
        self.predict_before_delivery = predict_before_delivery
        self.max_approval_hours = max_approval_hours
        self.max_carrier_hours = max_carrier_hours
        self.max_delivery_days = max_delivery_days
        self.df: Optional[pd.DataFrame] = None

    # --------------------------------------------------
    # Core sklearn methods
    # --------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.df = X.copy()

        self._parse_datetimes()
        self._create_target()
        self._create_time_features()
        self._clean_data()

        return self._get_final_dataset()

    # --------------------------------------------------
    # Internal Processing Steps
    # --------------------------------------------------

    def _parse_datetimes(self) -> None:
        for col in self.DATETIME_COLUMNS:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(
                    self.df[col],
                    errors="coerce"
                )
        logging.info("Datetime columns parsed safely.")

    def _create_target(self) -> None:
        if (
            "order_delivered_customer_date" in self.df.columns
            and "order_estimated_delivery_date" in self.df.columns
        ):
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

            logging.info(
                f"Target variable created. Shape: {self.df.shape}"
            )

    def _create_time_features(self) -> None:
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

        # Leakage-safe: only create total_delivery_days if allowed
        if not self.predict_before_delivery:
            self.df["total_delivery_days"] = (
                (self.df["order_delivered_customer_date"] - purchase)
                .dt.total_seconds() / (3600 * 24)
            )

        logging.info(
            f"Time-based features created. Shape: {self.df.shape}"
        )

    def _clean_data(self) -> None:
        self.df = self.df[
            self.df["approval_delay_hours"].between(
                0, self.max_approval_hours
            )
            & self.df["carrier_delay_hours"].between(
                0, self.max_carrier_hours
            )
            & self.df["estimated_delivery_days"].between(
                0, self.max_delivery_days
            )
        ]

        logging.info(
            f"Outliers removed. Shape: {self.df.shape}"
        )

    def merge_additional_data(
        self,
        payments_df: pd.DataFrame,
        customers_df: pd.DataFrame
    ) -> None:
        """
        Merge payment and customer datasets safely.
        """

        payment_agg = (
            payments_df
            .groupby("order_id")
            .agg(
                total_payment_value=("payment_value", "sum"),
                payment_installments=("payment_installments", "max")
            )
            .reset_index()
        )

        self.df = self.df.merge(
            payment_agg,
            on="order_id",
            how="left"
        )

        self.df = self.df.merge(
            customers_df[["customer_id", "customer_state"]],
            on="customer_id",
            how="left"
        )

        # Missing value handling
        self.df["total_payment_value"].fillna(0, inplace=True)
        self.df["payment_installments"].fillna(1, inplace=True)
        self.df["customer_state"].fillna("Unknown", inplace=True)

        logging.info(
            f"Additional datasets merged. Shape: {self.df.shape}"
        )

    def _get_final_dataset(self) -> pd.DataFrame:
        final_columns = self.FEATURE_COLUMNS.copy()

        if self.TARGET_COLUMN in self.df.columns:
            final_columns.append(self.TARGET_COLUMN)

        logging.info(
            f"Final feature set prepared. Columns: {final_columns}"
        )

        return self.df[final_columns]
