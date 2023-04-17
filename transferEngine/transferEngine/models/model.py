from typing import Tuple

import numpy as np
from keras import layers, models
from keras.models import Model


def encoder(input_shape: Tuple[int, int, int], bottleneck_size: int) -> Model:
    """Create a model for encoding images as a vector of size "bottleneck_size"."""
    inputs = layers.Input(shape=input_shape)

    # Main body of the processing
    x = layers.Conv2D(16, (5, 5), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (5, 5), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="LeakyReLU")(x)
    x = layers.Dropout(0.5)(x)

    # Skip connection
    skip = layers.Flatten()(inputs)

    # Combine the main body and the skip connection
    x = layers.concatenate([x, skip])
    x = layers.Dense(bottleneck_size, activation="LeakyReLU")(x)

    return Model(inputs=inputs, outputs=x)


def decoder(bottleneck_size: int) -> Model:
    """Create a model for decoding the bottleneck layer into an image."""
    inputs = layers.Input(shape=(bottleneck_size,))

    # Main body of the processing
    x = layers.Dense(1024, activation="LeakyReLU")(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Reshape((8, 8, 16))(x)
    x = layers.Conv2DTranspose(1024, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(x)

    return Model(inputs=inputs, outputs=x)


class AutoEncoder(Model):
    """An autoencoder model."""

    def __init__(self, input_shape):
        """Initialise the model with encoder and decoder layers."""
        super(AutoEncoder, self).__init__()
        self.encoder = encoder(input_shape, 1024)
        self.decoder = decoder(1024)

    def call(self, x):
        """Forward pass of the model."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def encode_and_combine(self, img_1: np.ndarray, img_2: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode and combine two images.

        Args:
            img_1: The first image to encode and combine.
            img_2: The second image to encode and combine.
            alpha: The split between the two images when combining.
        """
        encoded_1: np.ndarray = self.encoder(img_1)  # type: ignore
        encoded_2: np.ndarray = self.encoder(img_2)  # type: ignore

        combined = encoded_1 * alpha + encoded_2 * (1 - alpha)  # type: ignore

        decoded_1: np.ndarray = self.decoder(encoded_1)  # type: ignore
        decoded_2: np.ndarray = self.decoder(encoded_2)  # type: ignore
        decoded_combined: np.ndarray = self.decoder(combined)  # type: ignore

        return decoded_1, decoded_2, decoded_combined
