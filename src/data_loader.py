"""
Data Loader Module
Enterprise-Scale Predictive Analytics System

Responsible for:
- Loading raw datasets
- Validating file existence
- Parsing datetime columns
- Providing structured access to datasets
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class DataLoader:
    """
    Handles loading of Olist e-commerce datasets.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_path = Path(data_dir)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

    def _load_csv(self, filename: str, parse_dates: list | None = None) -> pd.DataFrame:
        file_path = self.data_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logging.info(f"Loading file: {filename}")
        df = pd.read_csv(file_path, parse_dates=parse_dates)
        logging.info(f"{filename} loaded successfully | Shape: {df.shape}")

        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets.
        Returns dictionary of DataFrames.
        """

        datasets = {
            "orders": self._load_csv(
                "olist_orders_dataset.csv",
                parse_dates=[
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date"
                ]
            ),
            "customers": self._load_csv("olist_customers_dataset.csv"),
            "order_items": self._load_csv("olist_order_items_dataset.csv"),
            "payments": self._load_csv("olist_order_payments_dataset.csv"),
            "reviews": self._load_csv("olist_order_reviews_dataset.csv"),
            "products": self._load_csv("olist_products_dataset.csv"),
            "sellers": self._load_csv("olist_sellers_dataset.csv"),
        }

        logging.info("All datasets loaded successfully.")

        return datasets
