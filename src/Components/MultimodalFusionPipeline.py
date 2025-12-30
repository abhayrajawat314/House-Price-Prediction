import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from .SatelliteArchitecture import SatelliteCNNEncoder
from .StreetArchitecture import StreetCNNEncoder


class MultimodalFusionPipeline:
    def __init__(
        self,
        train_csv_path,
        price_csv_path,
        tabular_encoder_path,
        tabular_scaler_path,
        tabular_features_path,
        satellite_encoder_path,
        street_encoder_path,
        satellite_image_dir,
        street_image_dir,
        missing_street_ids_csv
    ):
        self.train_csv_path = train_csv_path
        self.price_csv_path = price_csv_path

        self.tabular_encoder = load_model(tabular_encoder_path)
        self.satellite_encoder = load_model(satellite_encoder_path)
        self.street_encoder = load_model(street_encoder_path)

        self.scaler = joblib.load(tabular_scaler_path)
        self.feature_cols = joblib.load(tabular_features_path)

        self.satellite_image_dir = satellite_image_dir
        self.street_image_dir = street_image_dir
        self.missing_street_ids_csv = missing_street_ids_csv

    def generate_embeddings(self):
        df = pd.read_csv(self.train_csv_path)
        ids = df["id"].astype(str).str.strip().values

        X_tab = self.scaler.transform(df[self.feature_cols].values)
        tab_emb = self.tabular_encoder.predict(X_tab, verbose=0)

        sat_enc = SatelliteCNNEncoder(self.satellite_image_dir, self.train_csv_path)
        ids, sat_emb = sat_enc.generate_embeddings(self.satellite_encoder)

        street_enc = StreetCNNEncoder(
            self.street_image_dir,
            self.train_csv_path,
            self.price_csv_path,
            self.missing_street_ids_csv
        )
        _, street_emb = street_enc.generate_embeddings(self.street_encoder)

        return ids, tab_emb, sat_emb, street_emb

    def load_targets(self, ids):
        df = pd.read_csv(self.price_csv_path)
        id_to_y = dict(zip(df["id"].astype(str), df["log_price"]))
        return np.array([id_to_y[i] for i in ids], dtype="float32")
