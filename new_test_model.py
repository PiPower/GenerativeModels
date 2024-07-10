import math
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras import layers
"""
based on
https://keras.io/examples/generative/ddim/
"""
def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth, embbed_in = None):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
            if embbed_in is not None:
                embbed_scale = layers.Dense(width)(embbed_in)
                embbed_scale = tf.reshape(embbed_scale, (-1, 1, 1, width))
                x = x + embbed_scale
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth, embbed_in = None):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
            if embbed_in is not None:
                embbed_scale = layers.Dense(width)(embbed_in)
                embbed_scale = tf.reshape(embbed_scale, (-1, 1, 1, width))
                x = x + embbed_scale
        return x

    return apply


def get_network(image_size, widths, block_depth, embb, out_channels = 3, denoising_dm = True, offset_channels = False, embbed_dim = None):
    input_shape = (image_size, image_size, out_channels)
    if isinstance(image_size, tuple):
        input_shape = image_size + (out_channels, )
    noisy_images = keras.Input(shape=input_shape)
    noise_variances = keras.Input(shape=(embb))

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    if denoising_dm:
        e = layers.Dense(image_size* image_size* out_channels, activation='relu')(noise_variances)
        e = layers.Reshape( (image_size, image_size, out_channels )  )(e)
        x = layers.Concatenate()([x, e])

    embbed_in = layers.Dense(embbed_dim )(noise_variances) if offset_channels else None

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth, embbed_in)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth, embbed_in)([x, skips])

    x = layers.Conv2D(out_channels, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noise_variances, noisy_images], x, name="residual_unet")
