from src.Components.TabularArchitecture import TabularEncoder


encoder = TabularEncoder("Data/final_train_test_data/train_df.csv")
encoder.train_and_save(
    "artifacts/tabular_encoder.keras",
    "artifacts/tabular_scaler.pkl",
    "artifacts/tabular_features.pkl"
)
