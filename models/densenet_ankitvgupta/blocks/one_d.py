""" 
blocks/one_d.py
Author: Ankit Gupta

Implementations of various DenseNet blocks for 1D sequences

This module contains helper functions that define the various subcomponents of a DenseNet. This includes dense blocks and transition blocks.

"""

from keras.layers import BatchNormalization, Activation, Conv1D, Concatenate, AveragePooling1D


def H_l(k, bottleneck_size, kernel_width):
    """ 
    A single convolutional "layer" as defined by Huang et al. Defined as H_l in the original paper
    
    :param k: int representing the "growth rate" of the DenseNet
    :param bottleneck_size: int representing the size of the bottleneck, as a multiple of k. Set to 0 for no bottleneck.
    :param kernel_width: int representing the width of the main convolutional kernel
    :return a function wrapping the keras layers for H_l
    """

    use_bottleneck = bottleneck_size > 0
    num_bottleneck_output_filters = k * bottleneck_size

    def f(x):
        if use_bottleneck:
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(
                num_bottleneck_output_filters,
                1,
                strides=1,
                padding="same",
                dilation_rate=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(
            k,
            kernel_width,
            strides=1,
            padding="same",
            dilation_rate=1)(x)
        return x
    return f


def dense_block(k, num_layers, kernel_width, bottleneck_size):
    """
    A single dense block of the DenseNet
    
    :param k: int representing the "growth rate" of the DenseNet
    :param num_layers: int represending the number of layers in the block
    :param kernel_width: int representing the width of the main convolutional kernel
    :param bottleneck_size: int representing the size of the bottleneck, as a multiple of k. Set to 0 for no bottleneck.
    :return a function wrapping the entire dense block
    """
    def f(x):
        layers_to_concat = [x]
        for _ in range(num_layers):
            x = H_l(k, bottleneck_size, kernel_width)(x)
            layers_to_concat.append(x)
            x = Concatenate(axis=-1)(layers_to_concat)
        return x
    return f


def transition_block(pool_size=2, stride=2, theta=0.5):
    """
    A single transition block of the DenseNet
    
    :param pool_size: int represending the width of the average pool
    :param stride: int represending the stride of the average pool
    :param theta: int representing the amount of compression in the 1x1 convolution. Set to 1 for no compression.
    :return a function wrapping the entire transition block
    """    
    assert theta > 0 and theta <= 1

    def f(x):
        num_transition_output_filters = int(int(x.shape[2]) * float(theta))
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(
            num_transition_output_filters,
            1,
            strides=1,
            padding="same",
            dilation_rate=1)(x)
        x = AveragePooling1D(
            pool_size=pool_size,
            strides=stride,
            padding="same")(x)
        return x
    return f
