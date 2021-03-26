"""
models/one_d.py
Author: Ankit Gupta

Implementations of the core DenseNet 1D model

This module contains helper functions that define a DenseNet computational graph in Keras. Note that these functions are not immediately usable for classification, as the outputs are not softmaxed, and the functions have not been wrapped in keras.models.Model objects.
"""
from keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, GlobalAveragePooling1D
from ..blocks.one_d import dense_block, transition_block


def DenseNet(
        k,
        block_sizes,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling):
    def f(x):
        x = Conv1D(
            initial_filters,
            initial_conv_width,
            strides=initial_stride,
            padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling1D(
            pool_size=initial_pool_width,
            strides=initial_pool_stride,
            padding="same")(x)

        # Add all but the last dense block
        for block_size in block_sizes[:-1]:
            x = dense_block(
                k,
                block_size,
                conv_kernel_width,
                bottleneck_size)(x)
            x = transition_block(
                pool_size=transition_pool_size,
                stride=transition_pool_stride,
                theta=theta)(x)

        # Add the last dense block
        final_block_size = block_sizes[-1]
        x = dense_block(
            k,
            final_block_size,
            conv_kernel_width,
            bottleneck_size)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        if use_global_pooling:
            x = GlobalAveragePooling1D()(x)
        return x
    return f


def DenseNet121(
        k,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling):
    block_sizes = [6, 12, 24, 16]
    return DenseNet(
        k,
        block_sizes,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling)


def DenseNet169(
        k,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling):
    block_sizes = [6, 12, 32, 32]
    return DenseNet(
        k,
        block_sizes,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling)


def DenseNet201(
        k,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling):
    block_sizes = [6, 12, 48, 32]
    return DenseNet(
        k,
        block_sizes,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling)


def DenseNet264(
        k,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling):
    block_sizes = [6, 12, 64, 48]
    return DenseNet(
        k,
        block_sizes,
        conv_kernel_width,
        bottleneck_size,
        transition_pool_size,
        transition_pool_stride,
        theta,
        initial_conv_width,
        initial_stride,
        initial_filters,
        initial_pool_width,
        initial_pool_stride,
        use_global_pooling)
