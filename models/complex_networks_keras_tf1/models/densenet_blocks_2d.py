# -*- coding: utf-8 -*-

"""This module implements a number of popular two-dimensional complex valued DenseNet blocks."""

#  Authors: Qinggang Sun
#
#  Reference:
#       Somshubra Majumdar. DenseNet
#       https://github.com/titu1994/DenseNet

# pylint:disable=too-many-arguments, invalid-name, unused-variable

import keras.layers
import keras.regularizers

from ..layers.activations import layer_activation
from ..layers.bn import ComplexBatchNormalization
from ..layers.conv import ComplexConv2D
# from ..layers.conv import conv2d_transpose
from ..layers.pool import ComplexAveragePooling2D


def conv2d_block(inputs, nb_filter, activation='crelu', bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        inputs: keras tensor, complex valued inputs
        nb_filter: int, number of filters
        activation: char, activation function after the conventional layer
        bottleneck: bool, add bottleneck block
        dropout_rate: float, dropout rate
        weight_decay: float, weight decay factor
    Returns: keras tensor, with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

    x_complex = ComplexBatchNormalization(axis=concat_axis, epsilon=1.1e-5)(inputs)

    x_complex = layer_activation(x_complex, activation)

    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x_complex = ComplexConv2D(inter_channel, (1, 1), use_bias=False, spectral_parametrization=False,
                                  padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x_complex)

        x_complex = ComplexBatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x_complex)

        x_complex = layer_activation(x_complex, activation)

    x_complex = ComplexConv2D(inter_channel, (3, 3), use_bias=False, spectral_parametrization=False,
                              padding='same')(x_complex)

    if dropout_rate:
        x_complex = keras.layers.Dropout(dropout_rate)(x_complex)

    return x_complex


def dense2d_block(x_complex, nb_layers, nb_filter, growth_rate, activation='crelu', bottleneck=False,
                  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x_complex: keras tensor, inputs
        nb_layers: int, the number of layers of conv_block to append to the model.
        nb_filter: int, number of filters
        growth_rate: float, growth rate
        activation: char, activation function after the conventional layer
        bottleneck: bool, bottleneck block
        dropout_rate: float, dropout rate
        weight_decay: float, weight decay factor
        grow_nb_filters: bool, flag to decide to allow number of filters to grow
        return_concat_list: bool, return the list of feature maps along with the actual output
    Returns: keras tensor, with nb_layers of conv_block appended
    '''
    concat_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

    x_list = [x_complex]

    for i in range(nb_layers):
        cb = conv2d_block(x_complex, growth_rate, activation, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x_complex = keras.layers.concatenate([x_complex, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x_complex, nb_filter, x_list
    return x_complex, nb_filter


def transition2d_block(inputs, nb_filter, activation='crelu', compression=1.0, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        inputs: keras tensor
        nb_filter: number of filters
        activation: char, activation function after the conventional layer
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

    x_complex = ComplexBatchNormalization(axis=concat_axis, epsilon=1.1e-5)(inputs)

    x_complex = layer_activation(x_complex, activation)

    x_complex = ComplexConv2D(int(nb_filter * compression), (1, 1), use_bias=False, spectral_parametrization=False,
                              padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay))(x_complex)

    x_complex = ComplexAveragePooling2D((2, 2), strides=(2, 2))(x_complex)

    return x_complex


def transition2d_up_block(inputs, nb_filters, way='deconv', weight_decay=1e-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        inputs: keras tensor
        nb_filters: number of layers
        way: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    if way == 'upsampling':
        x_complex = keras.layers.UpSampling2D()(inputs)
    else:
        # x = keras.layers.Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same', strides=(2, 2),
        #                                  kernel_initializer='he_normal',
        #                                  kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)

        # x_complex = conv2d_transpose(inputs, nb_filters, (3, 3), strides=(2, 2))

        x_complex = ComplexConv2D(nb_filters, (3, 3), strides=(2, 2), padding='same', activation='crelu',
                                  transposed=True,
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)

    return x_complex
