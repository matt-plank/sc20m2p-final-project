import argparse
import logging
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from transferEngine.images import image_dataset_factory, image_factory
from transferEngine.images.image_dataset import ImageDataset
from transferEngine.models import model_factory


def plot_image_results(img_1, img_2, decoded_1, decoded_2, combined) -> Figure:
    """Plot the results of the model comparing the original images and the decoded images.

    Args:
        img_1: The first image.
        img_2: The second image.
        decoded_1: The first decoded image.
        decoded_2: The second decoded image.
        combined: The combined image.
    """
    # Create a 2x3 grid of plots
    fig: Figure
    axes: Any
    fig, axes = plt.subplots(2, 3)

    # Disable axes for all plots
    for ax in axes.flat:
        ax.axis("off")

    # Plot the original images on the first row
    axes[0, 0].imshow(img_1)
    axes[0, 0].set_title("Original Image 1")

    axes[0, 1].imshow(img_2)
    axes[0, 1].set_title("Original Image 2")

    # Plot the decoded images on the second row
    axes[1, 0].imshow(decoded_1)
    axes[1, 0].set_title("Decoded Image 1")

    axes[1, 1].imshow(decoded_2)
    axes[1, 1].set_title("Decoded Image 2")

    # Plot the combined image on the second row
    axes[1, 2].imshow(combined)
    axes[1, 2].set_title("Combined Image")

    return fig


def plot_training_history(training_history) -> Figure:
    """Plot the training history of the model.

    Args:
        training_history: The training history of the model.
    """
    # Create a new figure
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(111)  # type: ignore
    ax1.plot(training_history.history["accuracy"])
    ax1.plot(training_history.history["val_accuracy"])
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Training", "Validation"])

    return fig


def main():
    """The entry point of the program.

    This function will load the images, train the model, and plot the results.
    """
    # Set up logging & create a new logger
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger("Main")

    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--split", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--target-shape", type=int, nargs=3, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-save-path", type=str, required=True)
    args = parser.parse_args()

    IMAGE_SHAPE: Tuple[int, int, int] = tuple(args.target_shape)

    # Check if the dataset already exists
    dataset_path = Path(args.dataset_save_path)
    image_dataset: ImageDataset
    if dataset_path.exists():
        logger.info("Dataset already exists, loading...")
        image_dataset = image_dataset_factory.dataset_from_pickle(args.dataset_save_path)
    else:
        logger.info("Dataset does not exist, creating...")
        image_dataset = image_dataset_factory.dataset_from_path(
            args.dataset_path,
            target_size=IMAGE_SHAPE[:2],
            verbose=True,
        )
        image_dataset.save_to_pickle(args.dataset_save_path)

    # Train the model
    logger.info("Training model...")
    model, training_history = model_factory.create_and_train_model(
        IMAGE_SHAPE,
        image_dataset.images_as_matrix(),
        args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Demonstrate on some example images
    logger.info("Demonstrating on example images...")
    img_1 = image_factory.image_from_path("exampleImages/testImage.jpeg", IMAGE_SHAPE[:2]).wrapped_matrix
    img_2 = image_factory.image_from_path("exampleImages/testCombine.jpg", IMAGE_SHAPE[:2]).wrapped_matrix

    decoded_1, decoded_2, decoded_combined = model.encode_and_combine(img_1, img_2, 0.5)

    # Create and save plots of the results
    logger.info("Saving results...")
    plot_image_results(img_1[0], img_2[0], decoded_1[0], decoded_2[0], decoded_combined[0]).savefig(f"{args.model_path}/results.png")
    plot_training_history(training_history).savefig(f"{args.model_path}/accuracy.png")


if __name__ == "__main__":
    main()
