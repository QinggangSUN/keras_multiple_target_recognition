#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Qinggang Sun
#
# Reference:
#     Authors: Russell Doyosae
#     https://github.com/Doyosae/Deep_Complex_Networks

import numpy as np
import keras
import keras.backend as K
import tensorflow as tf


def complex_flatten (real, imag):

    real = keras.layers.Flatten()(real)
    imag = keras.layers.Flatten()(imag)

    return real, imag


def CReLU (real, imag):
    real = keras.layers.ReLU()(real)
    imag = keras.layers.ReLU()(imag)

    return real, imag


def zReLU (real, imag):

    real = keras.layers.ReLU()(real)
    imag = keras.layers.ReLU()(imag)

    real_flag = tf.cast(tf.cast(real, tf.bool), tf.float32)
    imag_flag = tf.cast(tf.cast(imag, tf.bool), tf.float32)

    flag = real_flag * imag_flag

    real = tf.math.multiply(real, flag)
    imag = tf.math.multiply(imag, flag)

    return real, imag


def modReLU (real, imag):

    norm = tf.abs(tf.complex(real, imag))
    bias = tf.Variable(np.zeros([norm.get_shape()[-1]]), trainable = True, dtype=tf.float32)
    relu = tf.nn.relu(norm + bias)

    real = tf.math.multiply(relu / norm + (1e+5), real)
    imag = tf.math.multiply(relu / norm + (1e+5), imag)

    return real, imag


def CLeaky_ReLU (real, imag):

    real = tf.nn.leaky_relu(real)
    imag = tf.nn.leaky_relu(imag)

    return real, imag


def complex_tanh (real, imag):

    real = tf.nn.tanh(real)
    imag = tf.nn.tanh(imag)

    return real, imag


def complex_softmax (real, imag):

    magnitude = tf.abs(tf.complex(real, imag))
    magnitude = keras.layers.Softmax()(magnitude)

    return complex_to_real_imag(magnitude)

def complex_real_sigmoid (real, imag):

    magnitude = tf.abs(tf.complex(real, imag))
    magnitude = keras.activations.sigmoid(magnitude)

    return complex_to_real_imag(magnitude)

def complex_to_real_imag(inputs):
    channel_axis = -1 if keras.backend.image_data_format() == "channels_last" else 1
    input_dim = K.int_shape(inputs)[channel_axis] // 2
    rank = K.ndim(inputs)
    if channel_axis == 1:
        if rank == 2:
            real = inputs[:, :input_dim]
            imag = inputs[:, input_dim:]
        elif rank == 3:
            real = inputs[:, :input_dim, :]
            imag = inputs[:, input_dim:, :]
        elif rank == 4:
            real = inputs[:, :input_dim, :, :]
            imag = inputs[:, input_dim:, :, :]
        elif rank == 5:
            real = inputs[:, :input_dim, :, :, :]
            imag = inputs[:, input_dim:, :, :, :]
    else:  # channels_axis == -1
        if rank == 2:
            real = inputs[:, :input_dim]
            imag = inputs[:, input_dim:]
        elif rank == 3:
            real = inputs[:, :, :input_dim]
            imag = inputs[:, :, input_dim:]
        elif rank == 4:
            real = inputs[:, :, :, :input_dim]
            imag = inputs[:, :, :, input_dim:]
        elif rank == 5:
            real = inputs[:, :, :, :, :input_dim]
            imag = inputs[:, :, :, :, input_dim:]

    return real, imag

def real_imag_to_complex(real, imag):
    channel_axis = -1 if keras.backend.image_data_format() == "channels_last" else 1
    output = keras.layers.concatenate([real, imag], axis=channel_axis)
    return output

_activation_dict = {'crelu':CReLU,
                    'zrelu':zReLU,
                    'mrelu':modReLU,
                    'clrelu':CLeaky_ReLU,
                    'ctanh':complex_tanh,
                    'csoftmax':complex_softmax,
                    'crsigmoid':complex_real_sigmoid}

def activation(inputs, key, input_form='complex'):
    if input_form == 'complex':  # ((,...,) + (input_dim*2=n_channels,))
        inputs = complex_to_real_imag(inputs)

    real, imag = inputs
    real, imag = _activation_dict[key](real, imag)
    output = real_imag_to_complex(real, imag)
    return output

def layer_activation(inputs, activation_key, input_form='complex', name=None):
    """A keras Lambda layer as activation layer."""
    outputs = keras.layers.Lambda(
            lambda inputs: (activation(inputs, activation_key, input_form)), name=name)(inputs)
    return outputs

