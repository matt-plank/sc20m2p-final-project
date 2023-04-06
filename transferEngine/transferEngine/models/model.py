from typing import Tuple

import numpy as np
from keras import layers, models
from keras.models import Model


class AutoEncoder(Model):
    """An autoencoder model."""

    def __init__(self, input_shape):
        """Initialise the model with encoder and decoder layers."""
        super(AutoEncoder, self).__init__()
        self.encoder = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
            ]
        )

        self.decoder = models.Sequential(
            [
                layers.Conv2DTranspose(512, (3, 3), activation="relu", padding="same"),
                layers.Conv2DTranspose(256, (3, 3), activation="relu", padding="same"),
                layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same"),
                layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same"),
                layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same"),
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dense(input_shape[0] * input_shape[1] * input_shape[2], activation="sigmoid"),
                layers.Reshape(input_shape),
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

        combined = encoded_1 * alpha + encoded_2 * (1 - alpha)

        decoded_1: np.ndarray = self.decoder(encoded_1)  # type: ignore
        decoded_2: np.ndarray = self.decoder(encoded_2)  # type: ignore
        decoded_combined: np.ndarray = self.decoder(combined)  # type: ignore

        return decoded_1, decoded_2, decoded_combined
