from typing import Callable, Tuple

import keras
import tensorflow as tf
from keras.applications import VGG16
from keras.losses import MeanAbsoluteError, MeanSquaredError


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def perceptual_loss(layer_names):
    vgg = VGG16(include_top=False, weights="imagenet", input_shape=(None, None, 3))
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]  # type: ignore
    model = keras.Model(inputs=vgg.input, outputs=outputs)

    def loss_function(y_true, y_pred):
        content_features = model(y_true)
        target_features = model(y_pred)

        loss = 0
        for content_feature, target_feature in zip(content_features, target_features):  # type: ignore
            loss += tf.reduce_mean(tf.square(content_feature - target_feature))

        return loss

    return loss_function


def combined_loss(*weights: float):
    perceptual_loss_function = perceptual_loss(("block3_conv3", "block4_conv3", "block5_conv3"))

    def loss(y_true, y_pred):
        mse = MeanSquaredError()(y_true, y_pred)
        mae = MeanAbsoluteError()(y_true, y_pred)
        ssim = ssim_loss(y_true, y_pred)
        perceptual = perceptual_loss_function(y_true, y_pred)
        return weights[0] * mse + weights[1] * mae + weights[2] * ssim + weights[3] * perceptual

    return loss
