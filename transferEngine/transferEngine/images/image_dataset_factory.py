"""Image dataset factory.

This module contains functions for creating image datasets.
All creation should be done through the functions in this module.
"""

import glob
import logging
import pickle
from pathlib import Path
from typing import Tuple

from . import image_factory
from .image_dataset import ImageDataset

logger = logging.getLogger("Image Dataset Factory")


def dataset_from_path(path: str, target_size: Tuple[int, int], verbose: bool = False) -> ImageDataset:
    """Load images from the directory at self.path.

    Args:
        target_size: The target size of the images.
        augment: Whether to augment the images by flipping and rotating them.
    """
    if verbose:
        logger.info(f"Loading images from {path}...")

    dataset = ImageDataset()

    # Find images recursively so they can be stored in subfolders by search term
    image_paths = glob.glob(f"{path}/**/*", recursive=True)
    image_paths = [path for path in image_paths if Path.is_file(Path(path))]

    for i, image_path in enumerate(image_paths):
        if verbose:
            logger.info(f"Loading {image_path} ({i + 1}/{len(image_paths)})")

        img = image_factory.image_from_path(image_path, target_size=target_size)
        dataset.images[image_path] = img

    return dataset


def dataset_from_pickle(path: str) -> ImageDataset:
    """Load an image dataset from a pickle file.

    Args:
        path: The path to the pickle file.
    """
    with open(path, "rb") as file:
        return pickle.load(file)
