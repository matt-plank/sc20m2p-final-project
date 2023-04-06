"""This module contains functions for loading images and creating image datasets.

Everything to do with images is contained in this module.
"""

import glob
from typing import Dict, List, Tuple

import numpy as np
from keras.utils import img_to_array, load_img
from skimage.transform import resize


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


def image_from_path(image_path: str, target_size: Tuple[int, int], expand: bool = False) -> Image:
    """Load an image into a numpy array.

    Args:
        image_path: The path to the image.
        target_size: The target size of the image.
        expand: Whether to wrap the image in an extra dimension (helpful for TensorFlow)
    """
    img = load_img(image_path)
    img = img_to_array(img)
    img = resize(img, target_size)
    img = img / 255.0  # type: ignore

    if expand:
        img = np.expand_dims(img, axis=0)

    return Image(img)


class ImageDataset:
    """A class for loading images from a directory and handling access to them."""

    def __init__(self, path: str):
        """Create a new ImageDataset ready to load images from the given path."""
        self.path = path
        self.images: Dict[str, Image] = {}

    def load_images(self, target_size: Tuple[int, int], augment: bool = False) -> None:
        """Load images from the directory at self.path.

        Args:
            target_size: The target size of the images.
            augment: Whether to augment the images by flipping and rotating them.
        """
        image_paths = glob.glob(f"{self.path}/*")

        for image_path in image_paths:
            img = image_from_path(image_path, target_size=target_size)
            self.images[image_path] = img

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
