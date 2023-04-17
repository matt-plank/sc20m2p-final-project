from typing import Tuple

import numpy as np
from keras import layers, models
from keras.models import Model


class Encoder(Model):
    def __init__(self, input_shape):
        """Initialise the encoder model.

        Args:
            input_shape: The shape of images the model will expect.
        """
        super(Encoder, self).__init__()
        self.encode_details = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2D(16, (5, 5), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(32, (5, 5), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Flatten(),
                layers.Dense(1024, activation="LeakyReLU"),
                layers.Dropout(0.5),
            ]
        )

        self.skip = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Flatten(),
            ]
        )

        self.encoder = models.Sequential(
            [
                layers.Dense(1024, activation="LeakyReLU"),
            ]
        )

    def call(self, x):
        """Forward pass of the model.

        Args:
            x: The input to the model.

        Returns:
            The encoded representation of the input.
        """
        processed = self.encode_details(x)
        skipped = self.skip(x)
        encoded = self.encoder(layers.concatenate([processed, skipped]))

        return encoded


class AutoEncoder(Model):
    """An autoencoder model."""

    def __init__(self, input_shape):
        """Initialise the model with encoder and decoder layers."""
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_shape)

        self.decoder = models.Sequential(
            [
                layers.Dense(1024, activation="LeakyReLU"),
                layers.Dropout(0.5),
                layers.Reshape((8, 8, 16)),
                layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2DTranspose(16, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2DTranspose(8, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.Conv2DTranspose(4, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

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
