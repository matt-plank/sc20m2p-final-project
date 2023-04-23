"""Model factory module.

This module contains functions for training and loading models.
All creation of models should be done through this module.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from keras.models import load_model

from .model import AutoEncoder


def create_model(input_shape: Tuple[int, int, int], optimizer, loss) -> AutoEncoder:
    """Create a new model and return it.

    Args:
        input_shape: The shape of the input to the model.
    """
    model: AutoEncoder = AutoEncoder(input_shape)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model


def create_or_load_model(path: str, input_shape: Tuple[int, int, int], optimizer, loss) -> AutoEncoder:
    """If a model exists at the given path, load it. Otherwise, create a new model and save it to the path.

    Args:
        path: The path to the model.
        input_shape: The shape of the input to the model.
    """
    try:
        model: AutoEncoder = load_model(path)  # type: ignore
    except OSError:
        model = create_model(input_shape, optimizer, loss)

    return model
