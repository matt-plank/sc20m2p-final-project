from typing import Tuple

import matplotlib.pyplot as plt

from transferEngine.images import image_dataset_factory
from transferEngine.models import model_factory

MODEL_PATH: str = "model.tf"
MODEL_TRAINING_EPOCHS: int = 200
MODEL_TRAINING_SPLIT: float = 0.2
MODEL_TRAINING_BATCH_SIZE: int = 16
IMAGE_SHAPE: Tuple[int, int, int] = (28, 28, 3)


def plot(img_1, img_2, decoded_1, decoded_2, decoded_combined, training_history):
    """Plot the results of the model.

    Args:
        img_1: The first image.
        img_2: The second image.
        decoded_1: The first decoded image.
        decoded_2: The second decoded image.
        decoded_combined: The combined decoded image.
        training_history: The training history of the model.
    """
    # Create a figure
    fig = plt.figure()
    plt.tight_layout()

    # Top row in 3x3 grid
    top_row = plt.subplot2grid((4, 3), (0, 0), colspan=3)
    blank_row = plt.subplot2grid((4, 3), (1, 0), colspan=3)
    first_image = plt.subplot2grid((4, 3), (2, 0))
    second_image = plt.subplot2grid((4, 3), (2, 1))
    decoded_first_image = plt.subplot2grid((4, 3), (3, 0))
    decoded_second_image = plt.subplot2grid((4, 3), (3, 1))
    decoded_combined_image = plt.subplot2grid((4, 3), (3, 2))
    empty_slot = plt.subplot2grid((4, 3), (2, 2))

    # Plot accuracy from training_history in the top row
    top_row.plot(training_history.history["accuracy"])
    top_row.plot(training_history.history["val_accuracy"])

    # Turn off axes for the middle row, last column
    empty_slot.axis("off")
    blank_row.axis("off")

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
    plt.savefig("model.tf/results.png")


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

    plot(img_1, img_2, decoded_1, decoded_2, decoded_combined, training_history)


if __name__ == "__main__":
    main()
