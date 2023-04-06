from typing import Dict, List

import numpy as np

from .image import Image


class ImageDataset:
    """A class for loading images from a directory and handling access to them."""

    def __init__(self):
        """Initialises the dataset with a dict for images."""
        self.images: Dict[str, Image] = {}

    def images_as_matrix(self, augment: bool = False) -> np.ndarray:
        """Return the images as a matrix.

        Args:
            augment: Whether to augment the images by flipping and rotating them.
        """
        result: List[np.ndarray] = []

        for image in self.images.values():
            result.append(image.matrix)

            if not augment:
                continue

            result.append(image.flipped(axis=0))

            for i in range(3):
                result.append(image.rotated(times=i))

        return np.array(result)
