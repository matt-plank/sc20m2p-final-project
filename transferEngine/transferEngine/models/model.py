from typing import Tuple

import numpy as np
from keras import layers
from keras.models import Model


class ResidualBlock(layers.Layer):
    """A residual block with two convolutional layers."""

    def __init__(self, filters: int, kernel_size: Tuple[int, int], activation: str):
        """Initialise the residual block with layers."""
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, activation=activation, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, activation=activation, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filters, kernel_size, activation=activation, padding="same")
        self.bn3 = layers.BatchNormalization()

    def call(self, x):
        """Run the layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        return layers.Add()([x, y])


class TransposeResidualBlock(layers.Layer):
    """A residual block with two transposed convolutional layers."""

    def __init__(self, filters: int, kernel_size: Tuple[int, int], activation: str):
        """Initialise the residual block with layers."""
        super(TransposeResidualBlock, self).__init__()
        self.conv1 = layers.Conv2DTranspose(filters, kernel_size, activation=activation, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(filters, kernel_size, activation=activation, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(filters, kernel_size, activation=activation, padding="same")
        self.bn3 = layers.BatchNormalization()

    def call(self, x):
        """Run the layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        y = self.conv2(x)
        y = self.bn2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        return layers.Add()([x, y])


def encoder(input_shape: Tuple[int, int, int]) -> Model:
    """Create a model for encoding images as a vector of size "bottleneck_size"."""
    inputs = layers.Input(shape=input_shape)

    # Main body of the processing
    x = ResidualBlock(64, (3, 3), "relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = ResidualBlock(128, (3, 3), "relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = ResidualBlock(192, (3, 3), "relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = ResidualBlock(12, (3, 3), "relu")(x)

    return Model(inputs=inputs, outputs=x)


def decoder(output_shape: Tuple[int, int, int]) -> Model:
    """Create a model for decoding the bottleneck layer into an image."""
    inputs = layers.Input(shape=(8, 8, 12))

    # Main body of the processing
    x = TransposeResidualBlock(96, (3, 3), "relu")(inputs)
    x = layers.UpSampling2D((2, 2))(x)

    x = TransposeResidualBlock(192, (3, 3), "relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = TransposeResidualBlock(128, (3, 3), "relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = TransposeResidualBlock(64, (3, 3), "relu")(x)
    x = TransposeResidualBlock(32, (3, 3), "relu")(x)

    result = layers.Conv2D(output_shape[-1], (3, 3), activation="sigmoid", padding="same")(x)

    return Model(inputs=inputs, outputs=result)


class AutoEncoder(Model):
    """An autoencoder model."""

    def __init__(self, input_shape):
        """Initialise the model with encoder and decoder layers."""
        super(AutoEncoder, self).__init__()
        self.encoder = encoder(input_shape)
        self.decoder = decoder(input_shape)

    def call(self, x):
        """Forward pass of the model."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def encode_and_combine(self, img_1: np.ndarray, img_2: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode and combine two images.

        Args:
            img_1: The first image to encode and combine.
            img_2: The second image to encode and combine.
            alpha: The split between the two images when combining.
        """
        encoded_1: np.ndarray = self.encoder(img_1)  # type: ignore
        encoded_2: np.ndarray = self.encoder(img_2)  # type: ignore

        combined = encoded_1 * alpha + encoded_2 * (1 - alpha)  # type: ignore

        decoded_1: np.ndarray = self.decoder(encoded_1)  # type: ignore
        decoded_2: np.ndarray = self.decoder(encoded_2)  # type: ignore
        decoded_combined: np.ndarray = self.decoder(combined)  # type: ignore

        return decoded_1, decoded_2, decoded_combined
