from typing import Tuple

import numpy as np
from keras.utils import img_to_array, load_img
from skimage.transform import resize

from .image import Image


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
