import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.utils import img_to_array, load_img
from skimage.transform import resize


def create_model() -> models.Sequential:
    """Create a model for re-creating an image."""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(128 * 128 * 3, activation="sigmoid"))
    model.add(layers.Reshape((128, 128, 3)))
    model.compile(loss="mse", optimizer="adam", metrics=["acc"])

    return model


def load_image(image_path: str) -> np.ndarray:
    """Load an image."""
    img = load_img(image_path)
    img = img_to_array(img)
    img = resize(img, (128, 128))

    return img


def main():
    # Train the model created by create_model() to re-create the image loaded by load_image()
    model = create_model()
    img = load_image("testImage.jpeg")
    img = resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    model.fit(img, img, epochs=10, batch_size=1)

    print(img)

    # Predict the image
    prediction = model.predict(img)

    # Plot the image
    plt.imshow(prediction[0])
    plt.show()


if __name__ == "__main__":
    main()
