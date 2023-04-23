"""Image factory module.

This module contains a function for creating images.
All creation of images should be done through this module.
"""

from typing import Tuple

import numpy as np
import PIL
import tensorflow as tf
from keras.utils import img_to_array, load_img

from .image import Image


def image_from_path(image_path: str, target_size: Tuple[int, int], expand: bool = False) -> Image:
    """Load an image into a numpy array.

    Args:
        image_path: The path to the image.
        target_size: The target size of the image.
        expand: Whether to wrap the image in an extra dimension (helpful for TensorFlow)
    """
    img: PIL.Image = load_img(image_path)  # type: ignore - VS Code doesn't seem to think PIL.Image is a thing
    img_as_array: np.ndarray = img_to_array(img)
    resized: tf.Tensor = tf.image.resize(img_as_array, target_size)
    resized_as_array: np.ndarray = resized.numpy()  # type: ignore - VS Code doesn know numpy() is a method
    result = resized_as_array / 255.0  # type: ignore

    if expand:
        result = np.expand_dims(img, axis=0)

    return Image(result)
