import glob
from typing import Tuple

import numpy as np


class Image:
    """A class for handling operations on images."""

    def __init__(self, matrix: np.ndarray) -> None:
        """Create a new Image from a numpy array."""
        self.matrix: np.ndarray = matrix

    @property
    def wrapped_matrix(self) -> np.ndarray:
        """Wrap the image in an extra dimension (helpful for TensorFlow).)"""
        return np.expand_dims(self.matrix, axis=0)

    def flipped(self, axis: int) -> np.ndarray:
        """Flip the image along the given axis."""
        return np.flip(self.matrix, axis=axis)

    def rotated(self, times: int) -> np.ndarray:
        """Rotate the image a given number of times."""
        img = self.matrix.copy()

        for _ in range(times):
            img = np.rot90(img)

        return img

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of the image."""
        return self.matrix.shape
