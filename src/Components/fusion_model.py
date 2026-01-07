import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, LayerNormalization, Dropout
)
from tensorflow.keras.models import Model


def build_fusion_model(
    tab_dim,
    sat_dim,
    fusion_hidden_dim=128
):
    tab_input = Input(shape=(tab_dim,), name="tabular_embedding")
    sat_input = Input(shape=(sat_dim,), name="satellite_embedding")

    tab = LayerNormalization()(tab_input)
    sat = LayerNormalization()(sat_input)

    fused = Concatenate(name="fusion_concat")([tab, sat])

    x = Dense(fusion_hidden_dim, activation="relu")(fused)
    x = Dropout(0.3)(x)

    x = Dense(96, activation="relu")(x)
    x = Dropout(0.25)(x)

    x = Dense(48, activation="relu")(x)
    x = Dropout(0.15)(x)

    output = Dense(1, name="log_price")(x)

    return Model(
        inputs=[tab_input, sat_input],
        outputs=output,
        name="fusion_model_tab_sat"
    )
