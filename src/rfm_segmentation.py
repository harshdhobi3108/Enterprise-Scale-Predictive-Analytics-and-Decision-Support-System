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
    # BUILD ENTERPRISE RFM TABLE
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
                Monetary=("payment_value", "sum"),
                First_Purchase=("order_purchase_timestamp", "min"),
                Last_Purchase=("order_purchase_timestamp", "max"),
                Customer_City=("customer_city", "first"),
                Customer_State=("customer_state", "first")
            )
            .reset_index()
        )

        # =====================================================
        # ENTERPRISE METRICS
        # =====================================================

        rfm["Customer_Tenure_Days"] = (
            rfm["Last_Purchase"] - rfm["First_Purchase"]
        ).dt.days

        rfm["Avg_Order_Value"] = (
            rfm["Monetary"] / rfm["Frequency"]
        )

        rfm["Revenue_Percentile"] = (
            rfm["Monetary"].rank(pct=True) * 100
        )

        rfm.fillna(0, inplace=True)

        return rfm

    # =====================================================
    # ENTERPRISE BUSINESS SEGMENTATION
    # =====================================================
    def segment(self, rfm_df, n_clusters=4):

        # Optional clustering layer (analytics only)
        rfm_df["Monetary_Log"] = np.log1p(rfm_df["Monetary"])
        rfm_df["Frequency_Log"] = np.log1p(rfm_df["Frequency"])

        features = ["Recency", "Frequency_Log", "Monetary_Log"]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(rfm_df[features])

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20
        )

        rfm_df["Cluster"] = kmeans.fit_predict(scaled_data)

        os.makedirs("models", exist_ok=True)
        joblib.dump(kmeans, "models/rfm_kmeans.pkl")
        joblib.dump(scaler, "models/rfm_scaler.pkl")

        # Apply enterprise business rules
        rfm_df = self._assign_segment_labels(rfm_df)

        return rfm_df

    # =====================================================
    # TRUE ENTERPRISE SEGMENT LOGIC
    # =====================================================
    def _assign_segment_labels(self, rfm_df):

        def classify(row):

            # LOST CUSTOMERS
            if row["Recency"] > 365:
                return "Lost Customers"

            # VIP CUSTOMERS
            if (
                row["Recency"] <= 180 and
                row["Frequency"] >= 3 and
                row["Revenue_Percentile"] >= 80
            ):
                return "VIP Customers"

            # LOYAL CUSTOMERS
            if (
                row["Recency"] <= 180 and
                row["Frequency"] >= 2
            ):
                return "Loyal Customers"

            # HIGH POTENTIAL (High revenue but low frequency)
            if (
                row["Revenue_Percentile"] >= 75 and
                row["Frequency"] == 1
            ):
                return "High Potential Customers"

            # NEW CUSTOMERS
            if row["Frequency"] == 1 and row["Recency"] <= 90:
                return "New Customers"

            return "Regular Customers"

        rfm_df["Segment"] = rfm_df.apply(classify, axis=1)

        # Strategic Tier Assignment
        rfm_df["Strategic_Tier"] = rfm_df["Segment"].map({
            "VIP Customers": "Core Revenue Driver",
            "Loyal Customers": "Growth Asset",
            "High Potential Customers": "Upside – Convert to Loyal",
            "New Customers": "Onboarding Phase",
            "Regular Customers": "Maintain Engagement",
            "Lost Customers": "Revenue At Risk"
        })

        return rfm_df

    # =====================================================
    # STRATEGY ENGINE
    # =====================================================
    @staticmethod
    def recommend_strategy(segment):

        strategies = {
            "VIP Customers":
                "Protect aggressively. Provide exclusive rewards and retention benefits.",
            "Loyal Customers":
                "Cross-sell, subscription plans, loyalty reinforcement.",
            "High Potential Customers":
                "Encourage repeat purchase and loyalty conversion.",
            "New Customers":
                "Strong onboarding, first-repeat incentives.",
            "Regular Customers":
                "Engagement campaigns and value reminders.",
            "Lost Customers":
                "Reactivation campaign and targeted offers."
        }

        return strategies.get(segment, "No strategy defined.")

    # =====================================================
    # REVENUE CONTRIBUTION
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