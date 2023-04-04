"""The model module. Contains the model and functions to train and use it.

Everything to do with the model is contained in this module.
"""

from typing import Tuple

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


def train_model(dataset: ImageDataset, split: float, epochs: int, batch_size: int) -> tuple:
    """Train a new model and return it.

    Args:
        dataset: The dataset to train the model on.
        split: The validation split to use when training the model.
    """
    model: AutoEncoder = AutoEncoder((128, 128, 3))
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    history = model.fit(
        dataset.image_matrix,
        dataset.image_matrix,
        validation_split=split,
        epochs=epochs,
        batch_size=batch_size,
    )

    return model, history


def train_or_load_model(path: str, dataset: ImageDataset, split: float, epochs: int, batch_size: int) -> tuple:
    """If a model exists at the given path, load it. Otherwise, train a new model and save it to the path.

    Args:
        path: The path to the model.
        dataset: The dataset to train the model on.
    """
    history = None

    try:
        model: AutoEncoder = load_model(path)  # type: ignore
    except OSError:
        model, history = train_model(dataset, split, epochs, batch_size)

    return model, history


def encode_and_combine(img_1: np.ndarray, img_2: np.ndarray, model: AutoEncoder, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
