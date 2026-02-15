from src.data_loader import DataLoader
from src.feature_engineering import DeliveryFeatureEngineer
from src.model_training import DeliveryDelayModelTrainer


def main():
    loader = DataLoader(data_dir="data/raw")
    data = loader.load_all()

    orders = data["orders"]
    payments = data["payments"]
    customers = data["customers"]

    feature_engineer = DeliveryFeatureEngineer(orders_df=orders)
    feature_engineer.create_target()
    feature_engineer.create_time_features()
    feature_engineer.clean_data()
    feature_engineer.merge_additional_data(
        payments_df=payments,
        customers_df=customers
    )

    processed_df = feature_engineer.get_processed_data()
    print("\nClass Distribution:")
    print(processed_df["is_delayed"].value_counts(normalize=True))


    trainer = DeliveryDelayModelTrainer(df=processed_df)
    trainer.train()
    trainer.evaluate()
    trainer.show_feature_importance()
    trainer.save_model()


if __name__ == "__main__":
    main()

