import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os


class RFMSegmenter:

    def __init__(self, orders, payments, customers):
        self.orders = orders.copy()
        self.payments = payments.copy()
        self.customers = customers.copy()

    # =====================================================
    # BUILD RFM TABLE
    # =====================================================
    def build_rfm(self):

        self.orders["order_purchase_timestamp"] = pd.to_datetime(
            self.orders["order_purchase_timestamp"]
        )

        payments_agg = (
            self.payments
            .groupby("order_id")
            .agg(payment_value=("payment_value", "sum"))
            .reset_index()
        )

        df = (
            self.orders
            .merge(payments_agg, on="order_id", how="left")
            .merge(self.customers, on="customer_id", how="left")
        )

        reference_date = df["order_purchase_timestamp"].max()

        rfm = (
            df.groupby("customer_unique_id")
            .agg(
                Recency=(
                    "order_purchase_timestamp",
                    lambda x: (reference_date - x.max()).days
                ),
                Frequency=("order_id", "nunique"),
                Monetary=("payment_value", "sum")
            )
            .reset_index()
        )

        return rfm

    # =====================================================
    # CLUSTER SEGMENTATION
    # =====================================================
    def segment(self, rfm_df, n_clusters=4):

        features = ["Recency", "Frequency", "Monetary"]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(rfm_df[features])

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20
        )

        rfm_df["Cluster"] = kmeans.fit_predict(scaled_data)

        # Save models
        os.makedirs("models", exist_ok=True)
        joblib.dump(kmeans, "models/rfm_kmeans.pkl")
        joblib.dump(scaler, "models/rfm_scaler.pkl")

        # Add business intelligence layer
        rfm_df = self._assign_segment_labels(rfm_df)

        return rfm_df

    # =====================================================
    # SEGMENT LABELING (BUSINESS LAYER)
    # =====================================================
    def _assign_segment_labels(self, rfm_df):

        cluster_summary = (
            rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .sort_values("Monetary", ascending=False)
        )

        cluster_order = cluster_summary.index.tolist()

        label_map = {}

        if len(cluster_order) >= 4:
            label_map[cluster_order[0]] = "VIP Customers"
            label_map[cluster_order[1]] = "Loyal Customers"
            label_map[cluster_order[2]] = "New Customers"
            label_map[cluster_order[3]] = "Lost Customers"
        else:
            for i, cluster in enumerate(cluster_order):
                label_map[cluster] = f"Segment {i}"

        rfm_df["Segment"] = rfm_df["Cluster"].map(label_map)

        return rfm_df

    # =====================================================
    # STRATEGY ENGINE
    # =====================================================
    @staticmethod
    def recommend_strategy(segment):

        strategies = {
            "VIP Customers":
                "Retention priority. Offer exclusive rewards & premium services.",
            "Loyal Customers":
                "Cross-sell, subscription incentives, and referral programs.",
            "New Customers":
                "Engage with onboarding campaigns and repeat-purchase incentives.",
            "Lost Customers":
                "Launch aggressive win-back campaigns and discount strategies."
        }

        return strategies.get(segment, "No strategy defined.")

    # =====================================================
    # REVENUE INTELLIGENCE
    # =====================================================
    @staticmethod
    def revenue_contribution(rfm_df):

        total_revenue = rfm_df["Monetary"].sum()

        segment_revenue = (
            rfm_df.groupby("Segment")["Monetary"]
            .sum()
            .reset_index()
        )

        segment_revenue["Revenue_%"] = (
            segment_revenue["Monetary"] / total_revenue * 100
        )

        return segment_revenue
