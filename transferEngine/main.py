import matplotlib.pyplot as plt
import numpy as np

from transferEngine.images import ImageDataset
from transferEngine.model import AutoEncoder, encode_and_combine, train_model

MODEL_PATH: str = "model.tf"


def plot(img_1, img_2, decoded_1, decoded_2, decoded_combined, training_history):
    """Plot the results of the model."""
    # Create a figure
    fig = plt.figure()
    plt.tight_layout()

    # Top row in 3x3 grid
    top_row = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    first_image = plt.subplot2grid((3, 3), (1, 0))
    second_image = plt.subplot2grid((3, 3), (1, 1))
    decoded_first_image = plt.subplot2grid((3, 3), (2, 0))
    decoded_second_image = plt.subplot2grid((3, 3), (2, 1))
    decoded_combined_image = plt.subplot2grid((3, 3), (2, 2))
    empty_slot = plt.subplot2grid((3, 3), (1, 2))

    # Plot accuracy from training_history in the top row
    top_row.plot(training_history.history["accuracy"])
    top_row.plot(training_history.history["val_accuracy"])

    # Turn off axes for the middle row, last column
    empty_slot.axis("off")

    # Show original images
    first_image.imshow(img_1[0])
    second_image.imshow(img_2[0])
    decoded_first_image.imshow(decoded_1[0])
    decoded_second_image.imshow(decoded_2[0])
    decoded_combined_image.imshow(decoded_combined[0])

    # Title each subplot
    top_row.set_title("Model Accuracy")
    top_row.set_ylabel("Accuracy")
    top_row.set_xlabel("Epoch")
    top_row.legend(["Train", "Validation"], loc="upper left")

    first_image.set_title("Original A")
    second_image.set_title("Original B")
    decoded_first_image.set_title("Decoded A")
    decoded_second_image.set_title("Decoded B")
    decoded_combined_image.set_title("Decoded Combined")

    # Plot decoded as an image
    plt.show()


def main():
    image_dataset = ImageDataset("trainingImages")
    image_dataset.load_images((32, 32), augment=True)

    model, training_history = train_model(image_dataset, 0.2)

    # Demonstrate on some example images
    img_1 = image_dataset.images["trainingImages/testImage.jpeg"]
    img_2 = image_dataset.images["trainingImages/testCombine.jpg"]

    decoded_1, decoded_2, decoded_combined = encode_and_combine(img_1, img_2, model, 0.5)

    plot(img_1, img_2, decoded_1, decoded_2, decoded_combined, training_history)


if __name__ == "__main__":
    main()
