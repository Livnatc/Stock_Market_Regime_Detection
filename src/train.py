import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler


# Convert data to sequences
def create_sequences(data, labels, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)


# Transformer Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs  # Residual connection

    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    return x + res  # Residual connection

def build_model(input_shape1, input_shape2):
    # Build Transformer Model
    input_layer = Input(shape=(input_shape1, input_shape2))
    x = transformer_encoder(input_layer, head_size=64, num_heads=4, ff_dim=128)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)

    # Pooling to reduce sequence dimension
    x = GlobalAveragePooling1D()(x)  # Converts (batch_size, 30, feature_dim) → (batch_size, feature_dim)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(3, activation="softmax")(x)  # 3 regimes: Bull, Bear, Sideways

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model