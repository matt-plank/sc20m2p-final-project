import matplotlib.pyplot as plt
import numpy as np

from transferEngine.images import ImageDataset
from transferEngine.model import AutoEncoder, encode_and_combine, train_or_load_model

MODEL_PATH: str = "model.tf"


def plot(img_1: np.ndarray, img_2: np.ndarray, decoded: np.ndarray, decoded_combine: np.ndarray, decoded_result: np.ndarray):
    """Plot the results of the model."""
    fig, axes = plt.subplots(2, 3)

    # Turn off axes for all subplots
    for ax in axes.flat:  # type: ignore
        ax.axis("off")

    # Show original images
    axes[0][0].imshow(img_1[0])
    axes[0][1].imshow(img_2[0])
    axes[1][0].imshow(decoded[0])
    axes[1][1].imshow(decoded_combine[0])
    axes[1][2].imshow(decoded_result[0])

    # Title each subplot
    axes[0][0].set_title("Original A")
    axes[0][1].set_title("Original B")
    axes[1][0].set_title("Decoded A")
    axes[1][1].set_title("Decoded B")
    axes[1][2].set_title("Decoded Combined")

    # Plot decoded as an image
    plt.show()


def main():
    image_dataset = ImageDataset("trainingImages")
    image_dataset.load_images((32, 32), augment=True)

    model: AutoEncoder = train_or_load_model(MODEL_PATH, image_dataset)

    # Demonstrate on some example images
    img_1 = image_dataset.images["trainingImages/testImage.jpeg"]
    img_2 = image_dataset.images["trainingImages/testCombine.jpg"]

    decoded_1, decoded_2, decoded_combined = encode_and_combine(img_1, img_2, model, 0.5)

    plot(img_1, img_2, decoded_1, decoded_2, decoded_combined)


if __name__ == "__main__":
    main()
