from src.Components.SatelliteArchitecture import SatelliteCNNEncoder

encoder = SatelliteCNNEncoder(
    "Data/train_images_CNN/satellite",
    "Data/final_train_test_data/train_df.csv",
    "Data/price/price_reference_train.csv"
)
encoder.train_and_save("artifacts/satellite_encoder.keras")
