import glob
from typing import Tuple

from . import image_factory
from .image_dataset import ImageDataset


def dataset_from_path(path: str, target_size: Tuple[int, int]) -> ImageDataset:
    """Load images from the directory at self.path.

    Args:
        target_size: The target size of the images.
        augment: Whether to augment the images by flipping and rotating them.
    """
    dataset = ImageDataset()

    image_paths = glob.glob(f"{path}/*")

    for image_path in image_paths:
        img = image_factory.image_from_path(image_path, target_size=target_size)
        dataset.images[image_path] = img

    return dataset
