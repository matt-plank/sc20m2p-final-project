from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def image_decode_figure(img, decoded) -> Figure:
    """Plot the results of the model comparing the original image and the decoded image.

    Args:
        img: The original image.
        decoded: The decoded image.
    """
    # Create a 1x2 grid of plots
    fig: Figure
    axes: Any
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Disable axes for all plots
    for ax in axes.flat:
        ax.axis("off")

    # Plot the original image on the first plot
    axes[0].imshow(img)
    axes[0].set_title("Original Image")

    # Plot the decoded image on the second plot
    axes[1].imshow(decoded)
    axes[1].set_title("Decoded Image")

    return fig


def latent_space_figure(feature_maps) -> Figure:
    """Plot a 3x4 grid of feature maps (grayscale).

    Each feature map is an 8x8 grid of values.
    """
    # Create a 3x4 grid of plots
    fig: Figure
    axes: Any
    fig, axes = plt.subplots(3, 4)

    # Disable axes for all plots
    for ax in axes.flat:
        ax.axis("off")

    # Plot each feature map
    for i in range(3):
        for j in range(4):
            index = i * 3 + j
            axes[i, j].imshow(feature_maps[index], cmap="gray")

    return fig


def image_results_figure(img_1, img_2, img_3, decoded_1, decoded_2, decoded_3, combined) -> Figure:
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
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    # Disable axes for all plots
    for ax in axes.flat:
        ax.axis("off")

    # Plot the original images on the first row
    axes[0, 0].imshow(img_1)
    axes[0, 0].set_title("Original Image 1")

    axes[0, 1].imshow(img_2)
    axes[0, 1].set_title("Original Image 2")

    axes[0, 2].imshow(img_3)
    axes[0, 2].set_title("Original Image 3")

    # Plot the decoded images on the second row
    axes[1, 0].imshow(decoded_1)
    axes[1, 0].set_title("Decoded Image 1")

    axes[1, 1].imshow(decoded_2)
    axes[1, 1].set_title("Decoded Image 2")

    axes[1, 2].imshow(decoded_3)
    axes[1, 2].set_title("Decoded Image 3")

    # Plot the combined image on the second row
    axes[1, 3].imshow(combined)
    axes[1, 3].set_title("Combined Image")

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
