import glob

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.utils import img_to_array, load_img
from skimage.transform import resize


def create_model() -> tuple[models.Sequential, models.Sequential, models.Sequential]:
    """Create a model for re-creating an image."""
    model = models.Sequential()

    # The layers of a tensorflow and keras model designed for processing 32x32 images
    encoder_layers = [
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
    ]

    # Re-create the image
    decoder_layers = [
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(32 * 32 * 3, activation="sigmoid"),
        layers.Reshape((32, 32, 3)),
    ]

    model = models.Sequential([*encoder_layers, *decoder_layers])
    encoder = models.Sequential(encoder_layers)
    decoder = models.Sequential(decoder_layers)

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model, encoder, decoder


def load_image(image_path: str, target_size: tuple[int, int], expand: bool = False) -> np.ndarray:
    """Load an image."""
    img = load_img(image_path)
    img = img_to_array(img)
    img = resize(img, target_size)
    img = img / 255.0

    if expand:
        img = np.expand_dims(img, axis=0)

    return img


def load_images(target_size: tuple[int, int], augment: bool = False) -> np.ndarray:
    """Load images from the trainingImages folder.

    Args:
        augment: Whether to augment the images by flipping and rotating them.
    """
    image_paths = glob.glob("trainingImages/*")

    images = []

    for image_path in image_paths:
        img = load_image(image_path, target_size=target_size)
        images.append(img)

        if not augment:
            continue

        images.append(np.flip(img, axis=0))

        for _ in range(3):
            img = np.rot90(img)

            images.append(img)
            images.append(np.flip(img, axis=0))
            images.append(np.flip(img, axis=1))

    return np.array(images)


def main():
    # Load the images
    img_1 = load_image("trainingImages/testImage.jpeg", (32, 32), expand=True)
    img_2 = load_image("trainingImages/testCombine.jpg", (32, 32), expand=True)

    training_images = load_images((32, 32), augment=True)

    # Create the model
    model, encoder, decoder = create_model()

    try:
        model.load_weights("model.h5")
    except OSError:
        model.fit(training_images, training_images, epochs=50, batch_size=32)
        model.save("model.h5")

    # Encode the images
    encoded = encoder.predict(img_1)
    encoded_combine = encoder.predict(img_2)

    combined = encoded * 0.5 + encoded_combine * 0.5

    # Decode the combined images
    result = decoder.predict(combined)

    # Plot decoded as an image
    plt.imshow(result[0])
    plt.show()


if __name__ == "__main__":
    main()
