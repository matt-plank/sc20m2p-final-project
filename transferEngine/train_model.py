from typing import Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from transferEngine.images import image_dataset_factory
from transferEngine.models import model_factory

MODEL_PATH: str = "model.tf"
MODEL_TRAINING_EPOCHS: int = 120
MODEL_TRAINING_SPLIT: float = 0.2
MODEL_TRAINING_BATCH_SIZE: int = 16
IMAGE_SHAPE: Tuple[int, int, int] = (28, 28, 3)


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
    image_dataset = image_dataset_factory.dataset_from_path(
        "trainingImages",
        target_size=IMAGE_SHAPE[:2],
    )

    model, training_history = model_factory.create_and_train_model(
        IMAGE_SHAPE,
        image_dataset.images_as_matrix(),
        MODEL_TRAINING_SPLIT,
        epochs=MODEL_TRAINING_EPOCHS,
        batch_size=MODEL_TRAINING_BATCH_SIZE,
    )

    # Demonstrate on some example images
    img_1 = image_dataset.images["trainingImages/testImage.jpeg"].wrapped_matrix
    img_2 = image_dataset.images["trainingImages/testCombine.jpg"].wrapped_matrix

    decoded_1, decoded_2, decoded_combined = model.encode_and_combine(img_1, img_2, 0.5)

    # Create and save plots of the results
    plot_image_results(img_1[0], img_2[0], decoded_1[0], decoded_2[0], decoded_combined[0]).savefig("model.tf/results.png")
    plot_training_history(training_history).savefig("model.tf/accuracy.png")


if __name__ == "__main__":
    main()
