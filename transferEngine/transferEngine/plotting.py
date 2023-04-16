from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def image_results_figure(img_1, img_2, decoded_1, decoded_2, combined) -> Figure:
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


def training_history_figure(training_history) -> Figure:
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
