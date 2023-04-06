from typing import Dict, Optional, Tuple

import numpy as np
from keras.models import load_model

from .model import AutoEncoder


def train_model(image_matrix: np.ndarray, split: float, epochs: int, batch_size: int) -> Tuple[AutoEncoder, Dict]:
    """Train a new model and return it.

    Args:
        dataset: The dataset to train the model on.
        split: The validation split to use when training the model.
        epochs: The number of epochs to train the model for.
        batch_size: The batch size to use when training the model.
    """
    model: AutoEncoder = AutoEncoder((128, 128, 3))
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    history = model.fit(
        image_matrix,
        image_matrix,
        validation_split=split,
        epochs=epochs,
        batch_size=batch_size,
    )

    return model, history


def train_or_load_model(path: str, image_matrix: np.ndarray, split: float, epochs: int, batch_size: int) -> Tuple[AutoEncoder, Optional[Dict]]:
    """If a model exists at the given path, load it. Otherwise, train a new model and save it to the path.

    Args:
        path: The path to the model.
        dataset: The dataset to train the model on.
    """
    history = None

    try:
        model: AutoEncoder = load_model(path)  # type: ignore
    except OSError:
        model, history = train_model(image_matrix, split, epochs, batch_size)

    return model, history
