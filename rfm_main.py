from src.data_loader import DataLoader
from src.rfm_segmentation import RFMSegmenter


def main():

    loader = DataLoader(data_dir="data/raw")
    data = loader.load_all()

    orders = data["orders"]
    payments = data["payments"]
    customers = data["customers"]

    segmenter = RFMSegmenter(orders, payments, customers)

    rfm_df = segmenter.build_rfm()
    segmented_df = segmenter.segment(rfm_df, n_clusters=4)

    print("\nCluster Distribution:")
    print(segmented_df["Cluster"].value_counts())

    print("\nCluster Summary:")
    print(
        segmented_df.groupby("Cluster")[
            ["Recency", "Frequency", "Monetary"]
        ].mean()
    )


if __name__ == "__main__":
    main()
