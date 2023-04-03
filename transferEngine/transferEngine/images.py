"""This module contains functions for loading images and creating image datasets.

Everything to do with images is contained in this module.
"""

import glob
from typing import Dict, Tuple

import numpy as np
from keras.utils import img_to_array, load_img
from skimage.transform import resize


def load_image(image_path: str, target_size: Tuple[int, int], expand: bool = False) -> np.ndarray:
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

    return img


class ImageDataset:
    """A class for loading images from a directory and handling access to them."""

    def __init__(self, path: str):
        """Create a new ImageDataset ready to load images from the given path."""
        self.path = path
        self.images: Dict = {}
        self.image_matrix: np.ndarray | None = None

    def load_images(self, target_size: Tuple[int, int], augment: bool = False) -> None:
        """Load images from the directory at self.path.

        Args:
            target_size: The target size of the images.
            augment: Whether to augment the images by flipping and rotating them.
        """
        image_paths = glob.glob(f"{self.path}/*")

        images = []

        for image_path in image_paths:
            img = load_image(image_path, target_size=target_size)
            images.append(img)

            self.images[image_path] = np.expand_dims(img, axis=0)

            if not augment:
                continue

            images.append(np.flip(img, axis=0))

            for _ in range(3):
                img = np.rot90(img)

                images.append(img)
                images.append(np.flip(img, axis=0))
                images.append(np.flip(img, axis=1))

        self.image_matrix = np.array(images)
