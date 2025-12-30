import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Lambda,
    LayerNormalization, Dropout
)
from tensorflow.keras.models import Model


def compute_mask(x):
    return tf.cast(
        tf.reduce_any(tf.not_equal(x, 0.0), axis=1, keepdims=True),
        tf.float32
    )


def apply_mask(inputs):
    x, mask = inputs
    return tf.stop_gradient(x) * mask



def build_fusion_model(
    tab_dim,
    sat_dim,
    street_dim,
    fusion_hidden_dim=192
):

    tab_input = Input(shape=(tab_dim,), name="tabular_embedding")
    sat_input = Input(shape=(sat_dim,), name="satellite_embedding")
    street_input = Input(shape=(street_dim,), name="street_embedding")

    
    tab = LayerNormalization()(tab_input)
    sat = LayerNormalization()(sat_input)
    street = LayerNormalization()(street_input)

   
    sat_mask = Lambda(
        compute_mask,
        output_shape=(1,),
        name="sat_mask"
    )(sat)

    street_mask = Lambda(
        compute_mask,
        output_shape=(1,),
        name="street_mask"
    )(street)

    
    sat_masked = Lambda(
        apply_mask,
        output_shape=lambda s: s[0],
        name="sat_masked"
    )([sat, sat_mask])

    street_masked = Lambda(
        apply_mask,
        output_shape=lambda s: s[0],
        name="street_masked"
    )([street, street_mask])

    
    fused = Concatenate(name="fusion_concat")(
        [tab, sat_masked, street_masked]
    )

    
    x = Dense(fusion_hidden_dim, activation="relu")(fused)
    x = Dropout(0.3)(x)

    x = Dense(96, activation="relu")(x)
    x = Dropout(0.25)(x)

    x = Dense(48, activation="relu")(x)
    x = Dropout(0.15)(x)

    output = Dense(1, name="log_price")(x)

    return Model(
        inputs=[tab_input, sat_input, street_input],
        outputs=output,
        name="multimodal_fusion_model"
    )
