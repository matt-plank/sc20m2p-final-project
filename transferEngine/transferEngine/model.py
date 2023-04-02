"""The model module. Contains the model and functions to train and use it.

Everything to do with the model is contained in this module.
"""

import numpy as np
from keras import layers, models
from keras.models import Model, load_model

from .images import ImageDataset


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
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
            ]
        )

        self.decoder = models.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(256, activation="relu"),
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


def train_or_load_model(path: str, dataset: ImageDataset) -> AutoEncoder:
    """If a model exists at the given path, load it. Otherwise, train a new model and save it to the path.

    Args:
        path: The path to the model.
        dataset: The dataset to train the model on.
    """
    try:
        model: AutoEncoder = load_model(path)  # type: ignore
    except OSError:
        model: AutoEncoder = AutoEncoder((32, 32, 3))
        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        model.fit(dataset.image_matrix, dataset.image_matrix, epochs=100, batch_size=32)
        model.save(path, save_format="tf")

    return model


def encode_and_combine(img_1: np.ndarray, img_2: np.ndarray, model: AutoEncoder, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode and combine two images using a model.

    Args:
        img_1: The first image to encode and combine.
        img_2: The second image to encode and combine.
        model: The model to use to encode and combine the images.
        alpha: The split between the two images when combining.
    """
    encoded_1: np.ndarray = model.encoder(img_1)  # type: ignore
    encoded_2: np.ndarray = model.encoder(img_2)  # type: ignore

    combined = encoded_1 * alpha + encoded_2 * (1 - alpha)

    decoded_1: np.ndarray = model.decoder(encoded_1)  # type: ignore
    decoded_2: np.ndarray = model.decoder(encoded_2)  # type: ignore
    decoded_combined: np.ndarray = model.decoder(combined)  # type: ignore

    return decoded_1, decoded_2, decoded_combined
