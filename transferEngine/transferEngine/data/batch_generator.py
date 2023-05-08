import glob
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
from PIL import Image


def is_image(path_str: str) -> bool:
    """Check if a file is an image.

    Args:
        path: The path to the file.
    """
    path = Path(path_str)

    if not path.is_file():
        return False

    return path.suffix.lower() in [".jpg", ".jpeg", ".png"]


def images_from_dir(path: str) -> List[str]:
    """Load images from a directory.

    Args:
        path: The path to the directory.

    Returns:
        A list of image paths.
    """
    # Recursively find all images in the directory
    image_paths: List[str] = glob.glob(f"{path}/**/*", recursive=True)
    image_paths = list(filter(is_image, image_paths))

    return image_paths


def from_paths(image_paths: List[str], batch_size: int) -> Generator:
    """Load a batch of images from a directory.

    Args:
        image_paths: The paths to the images.
        batch_size: The size of the batch.

    Returns:
        A generator that yields batches of images.
    """
    images = []

    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        images.append(img)

        if img.shape != (64, 64, 3):
            raise Exception(f"Image {image_path} has shape {img.shape}")

        if len(images) == batch_size:
            result = np.array(images, dtype=np.float32)
            result /= 255.0
            yield result, result
            images = []


def split_images(image_paths: List[str], split: float) -> Tuple[List[str], List[str]]:
    """Split a list of images into training and validation sets.

    Args:
        image_paths: The paths to the images.
        split: The proportion of images to use for validation.

    Returns:
        A tuple containing the training and validation sets.
    """
    num_validation = int(len(image_paths) * split)
    validation_image_paths = image_paths[:num_validation]
    training_image_paths = image_paths[num_validation:]

    return training_image_paths, validation_image_paths
