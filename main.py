import glob

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.utils import img_to_array, load_img
from skimage.transform import resize


def create_model() -> tuple[models.Sequential, models.Sequential, models.Sequential]:
    """Create a model for re-creating an image."""
    model = models.Sequential()
    encoder_layers = [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
    ]

    # Re-create the image
    decoder_layers = [
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(128 * 128 * 3, activation="sigmoid"),
        layers.Reshape((128, 128, 3)),
    ]

    model = models.Sequential([*encoder_layers, *decoder_layers])
    encoder = models.Sequential(encoder_layers)
    decoder = models.Sequential(decoder_layers)

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model, encoder, decoder


def load_image(image_path: str, expand: bool = False) -> np.ndarray:
    """Load an image."""
    img = load_img(image_path)
    img = img_to_array(img)
    img = resize(img, (128, 128))
    img = img / 255.0

    if expand:
        img = np.expand_dims(img, axis=0)

    return img


def load_images() -> np.ndarray:
    """Load images from a directory."""
    image_paths = glob.glob("trainingImages/*")

    images = []

    for image_path in image_paths:
        img = load_image(image_path)
        images.append(img)
        images.append(np.flip(img, axis=0))
        images.append(np.flip(img, axis=1))

        for _ in range(3):
            img = np.rot90(img)

            images.append(img)
            images.append(np.flip(img, axis=0))
            images.append(np.flip(img, axis=1))

    return np.array(images)


def main():
    # Load the images
    img_1 = load_image("trainingImages/testImage.jpeg", expand=True)
    img_2 = load_image("trainingImages/testCombine.jpg", expand=True)

    training_images = load_images()

    # Create the model
    model, encoder, decoder = create_model()

    try:
        model.load_weights("model.h5")
    except OSError:
        model.fit(training_images, training_images, epochs=5, batch_size=1)
        model.save("model.h5")

    # Encode the images
    encoded = encoder.predict(img_1)
    encoded_combine = encoder.predict(img_2)

    combined = (encoded + encoded_combine) / 2

    # Decode the combined images
    result = decoder.predict(combined)

    # Plot decoded as an image
    plt.imshow(result[0])
    plt.show()


if __name__ == "__main__":
    main()
