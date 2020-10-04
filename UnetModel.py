
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Dropout,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Flatten
)


# MODEL ARCHITECTURE
n_channels = 6
input_size = 256

num_layers = 2
input_shape = (input_size, input_size, n_channels)


def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x


def UnetModel(input_shape, filters, upconv_filters, num_layers):
    inputs = Input(input_shape)
    kernel_size = (3, 3)
    activation = 'relu'
    strides = (1, 1)
    padding = 'same'
    kernel_initializer = 'he_normal'
    output_activation = 'sigmoid'

    conv2d_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': strides,
        'padding': padding,
        'kernel_initializer': kernel_initializer
    }

    conv2d_trans_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': (2, 2),
        'padding': padding,
    }

    bachnorm_momentum = 0.01

    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_padding = 'valid'

    maxpool2d_args = {
        'pool_size': pool_size,
        'strides': pool_strides,
        'padding': pool_padding,
    }

    x = Conv2D(filters, **conv2d_args)(inputs)
    c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
    x = MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    for conv in reversed(down_layers):
        x = concatenate([x, conv])
        x = bn_conv_relu(
            x, upconv_filters, bachnorm_momentum, **conv2d_args
        )
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_upconv_relu(
            x, filters, bachnorm_momentum, **conv2d_trans_args
        )

    x = concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)

    outputs = Conv2D(
        1,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=output_activation,
        padding='valid')(x)

    return Model(inputs=[inputs], outputs=[outputs])
