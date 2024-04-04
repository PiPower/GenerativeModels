import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout, Conv2D, MaxPool2D, Conv2DTranspose, Input, Embedding, Dense, Reshape, BatchNormalization
import tensorflow_addons as tfa
import math
import os


def double_conv_block(x, n_filters, dropout_rate, ):

    # Conv2D then ReLU activation
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu")(x)
    x = Dropout(dropout_rate)(x)
    # Conv2D then ReLU activation
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu")(x)
    x = Dropout(dropout_rate)(x)

    return x

def downsample_block(x,t, n_filters, dropout_rate):
    x_old = tf.concat([x, t], axis=-1)
    x_old = Conv2D(n_filters, 3, 2,  padding="same")(x_old)
    x = double_conv_block(x_old, n_filters, dropout_rate)

    return x + x_old

def upsample_block(x, t,skip, n_filters, dropout_rate):
    # upsample
    x = tf.concat([x + skip, t], axis=-1)
    #x = x + t
    x_old = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = Dropout(dropout_rate)(x_old)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters, dropout_rate)
    return x + x_old

def build_unet_model_test(shape,output_channels, num_steps, hidden_dims, dropout_rate=0.1, att_block = -1, embedd_dim  = 64):
    # inputs
    num_blocks = len(hidden_dims) - 1
    embedding_input = Input(shape=(embedd_dim))
    inputs =Input(shape=shape)

    x = inputs
    skips = []

    x_upward = double_conv_block(x, hidden_dims[0], dropout_rate)
    skips.append(x_upward)
    x_downward = double_conv_block(x, hidden_dims[0], dropout_rate) + x_upward
    skips.append(x_downward)

    for i in range(1,num_blocks+1):
        t_in = Dense( shape[0] // 2 ** (i-1) * shape[1] // 2 ** (i-1) * 1, activation = 'relu' )(embedding_input)
        t_in = Reshape((shape[0] // 2 ** (i-1), shape[1] // 2 ** (i-1), 1))(t_in)

        x_upward = BatchNormalization(center=False, scale=False)(x_upward)
        x_downward =BatchNormalization(center=False, scale=False)(x_downward)

        x_upward = downsample_block(x_upward, t_in, hidden_dims[i], dropout_rate)
        skips.append(x_upward)
        x_downward = downsample_block(x_downward, t_in, hidden_dims[i], dropout_rate) + x_upward
        skips.append(x_downward)

    for i in range(num_blocks ,0 ,-1):
        t_in = Dense(  shape[0] // 2 ** (i) * shape[1] // 2 ** (i) * 1, activation = 'relu' )(embedding_input)
        t_in = Reshape((shape[0] // 2 ** (i), shape[1] // 2 ** (i), 1))(t_in)

        x_upward = BatchNormalization(center=False, scale=False)(x_upward)
        x_downward =BatchNormalization(center=False, scale=False)(x_downward)

        x_upward = upsample_block(x_upward, t_in, skips[2*i ], hidden_dims[i - 1],dropout_rate )
        x_downward = upsample_block(x_downward, t_in, skips[2*i +1 ], hidden_dims[i- 1],dropout_rate ) + x_upward

    x_upward_out = double_conv_block(x_upward, hidden_dims[0], dropout_rate) + skips[0]
    x_downward_out = double_conv_block(x_downward, hidden_dims[0], dropout_rate) + skips[1] + x_upward_out

    outputs = Conv2D(output_channels, 3, padding="same")(x_downward_out)
    unet_model = tf.keras.Model(inputs=[embedding_input, inputs], outputs=outputs, name="U-Net")

    return unet_model

