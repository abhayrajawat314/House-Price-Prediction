from src.Components.StreetArchitecture import StreetCNNEncoder

encoder = StreetCNNEncoder(
    "Data/train_images_CNN/street",
    "Data/final_train_test_data/train_df.csv",
    "Data/price/price_reference_train.csv",
    "Data/train_images_CNN/missing_street_ids.csv"
)
encoder.train_and_save("artifacts/street_encoder.keras")
