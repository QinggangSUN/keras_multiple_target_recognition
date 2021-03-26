#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Complex valued pooling layer."""

# Authors: Qinggang Sun
#
# Reference:
#     Authors: Olexa Bilaniuk
#     https://github.com/ChihebTrabelsi/deep_complex_networks
#
#     Authors: Dramsch Jesper
#     https://github.com/zengjie617789/keras-complex
#

# pylint:disable=too-many-arguments, invalid-name, bad-whitespace, inconsistent-return-statements
# pylint:disable=no-else-return, arguments-differ, redefined-outer-name, abstract-method, c-extension-no-member

import keras.backend                        as KB
import keras.engine                         as KE
import keras.layers                         as KL
import numpy                                as np
from .activations import complex_to_real_imag, real_imag_to_complex

#
# Spectral Pooling Layer
#

class SpectralPooling1D(KL.Layer):
    """SpectralPooling1D"""
    def __init__(self, topf=None, gamma=None, **kwargs):
        super(SpectralPooling1D, self).__init__(**kwargs)
        if topf:
            self.topf  = (int(topf[0]),)
            self.topf  = (self.topf[0]//2,)
        elif gamma:
            self.gamma = (float(gamma[0]),)
            self.gamma = (self.gamma[0]/2,)
        else:
            raise RuntimeError("Must provide either topf= or gamma= !")

    def call(self, x, mask=None):
        xshape = KB.int_shape(x)
        if hasattr(self, "topf"):
            topf = self.topf
        else:
            if KB.image_data_format() == "channels_first":
                topf = (int(self.gamma[0]*xshape[2]),)
            else:
                topf = (int(self.gamma[0]*xshape[1]),)

        if KB.image_data_format() == "channels_first":
            if topf[0] > 0 and xshape[2] >= 2*topf[0]:
                mask = [1]*(topf[0]              ) +\
                    [0]*(xshape[2] - 2*topf[0]) +\
                    [1]*(topf[0]              )
                mask = [[mask]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,2))
                mask = KB.constant(mask)
                x   *= mask
        else:
            if topf[0] > 0 and xshape[1] >= 2*topf[0]:
                mask = [1]*(topf[0]              ) +\
                    [0]*(xshape[1] - 2*topf[0]) +\
                    [1]*(topf[0]              )
                mask = [[mask]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,2,1))
                mask = KB.constant(mask)
                x   *= mask

        return x

class SpectralPooling2D(KL.Layer):
    """SpectralPooling2D"""
    def __init__(self, topf=None, gamma=None, **kwargs):
        super(SpectralPooling2D, self).__init__(**kwargs)
        if topf:
            self.topf  = (int  (topf[0]), int  (topf[1]))
            self.topf  = (self.topf[0]//2, self.topf[1]//2)
        elif gamma:
            self.gamma = (float(gamma[0]), float(gamma[1]))
            self.gamma = (self.gamma[0]/2, self.gamma[1]/2)
        else:
            raise RuntimeError("Must provide either topf= or gamma= !")

    def call(self, x, mask=None):
        xshape = KB.int_shape(x)
        if hasattr(self, "topf"):
            topf = self.topf
        else:
            if KB.image_data_format() == "channels_first":
                topf = (int(self.gamma[0]*xshape[2]), int(self.gamma[1]*xshape[3]))
            else:
                topf = (int(self.gamma[0]*xshape[1]), int(self.gamma[1]*xshape[2]))

        if KB.image_data_format() == "channels_first":
            if topf[0] > 0 and xshape[2] >= 2*topf[0]:
                mask = [1]*(topf[0]              ) +\
                    [0]*(xshape[2] - 2*topf[0]) +\
                    [1]*(topf[0]              )
                mask = [[[mask]]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,3,2))
                mask = KB.constant(mask)
                x   *= mask
            if topf[1] > 0 and xshape[3] >= 2*topf[1]:
                mask = [1]*(topf[1]              ) +\
                    [0]*(xshape[3] - 2*topf[1]) +\
                    [1]*(topf[1]              )
                mask = [[[mask]]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,2,3))
                mask = KB.constant(mask)
                x   *= mask
        else:
            if topf[0] > 0 and xshape[1] >= 2*topf[0]:
                mask = [1]*(topf[0]              ) +\
                    [0]*(xshape[1] - 2*topf[0]) +\
                    [1]*(topf[0]              )
                mask = [[[mask]]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,3,1,2))
                mask = KB.constant(mask)
                x   *= mask
            if topf[1] > 0 and xshape[2] >= 2*topf[1]:
                mask = [1]*(topf[1]              ) +\
                    [0]*(xshape[2] - 2*topf[1]) +\
                    [1]*(topf[1]              )
                mask = [[[mask]]]
                mask = np.asarray(mask, dtype=KB.floatx()).transpose((0,1,3,2))
                mask = KB.constant(mask)
                x   *= mask

        return x


class _ComplexPooling(KL.Layer):
    """Abstract class for different complex pooling layers.
    """

    def __init__(self, pool_size, strides, padding, data_format, **kwargs):
        super(_ComplexPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        raise NotImplementedError

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(_ComplexPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class _ComplexPooling1D(_ComplexPooling):
    """Abstract class for different complex pooling 1D layers.
    """

    def __init__(self, pool_size=2, strides=None,
                 padding='valid', data_format='channels_last', **kwargs):
        super(_ComplexPooling1D, self).__init__(pool_size, strides, padding,
                                                data_format, **kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = KL.conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
        self.strides = KL.conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = KL.conv_utils.normalize_padding(padding)
        self.data_format = KB.normalize_data_format(data_format)
        self.input_spec = KE.base_layer.InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            steps = input_shape[2]
            features = input_shape[1]
        else:
            steps = input_shape[1]
            features = input_shape[2]
        length = KL.conv_utils.conv_output_length(steps,
                                                  self.pool_size[0],
                                                  self.padding,
                                                  self.strides[0])
        if self.data_format == 'channels_first':
            return (input_shape[0], features, length)
        else:
            return (input_shape[0], length, features)

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(_ComplexPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class _ComplexPooling2D(_ComplexPooling):
    """Abstract class for different complex pooling 1D layers.
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(_ComplexPooling2D, self).__init__(pool_size, strides, padding,
                                                data_format, **kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = KL.conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = KL.conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = KL.conv_utils.normalize_padding(padding)
        self.data_format = KB.normalize_data_format(data_format)
        self.input_spec = KE.base_layer.InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = KL.conv_utils.conv_output_length(rows, self.pool_size[0],
                                                self.padding, self.strides[0])
        cols = KL.conv_utils.conv_output_length(cols, self.pool_size[1],
                                                self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(_ComplexPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class _ComplexPooling3D(_ComplexPooling):
    """Abstract class for different complex pooling 1D layers.
    """

    def __init__(self, pool_size=(2, 2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(_ComplexPooling3D, self).__init__(pool_size, strides, padding,
                                                data_format, **kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = KL.conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
        self.strides = KL.conv_utils.normalize_tuple(strides, 3, 'strides')
        self.padding = KL.conv_utils.normalize_padding(padding)
        self.data_format = KB.normalize_data_format(data_format)
        self.input_spec = KE.base_layer.InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            len_dim1 = input_shape[2]
            len_dim2 = input_shape[3]
            len_dim3 = input_shape[4]
        elif self.data_format == 'channels_last':
            len_dim1 = input_shape[1]
            len_dim2 = input_shape[2]
            len_dim3 = input_shape[3]
        len_dim1 = KL.conv_utils.conv_output_length(len_dim1, self.pool_size[0],
                                                    self.padding, self.strides[0])
        len_dim2 = KL.conv_utils.conv_output_length(len_dim2, self.pool_size[1],
                                                    self.padding, self.strides[1])
        len_dim3 = KL.conv_utils.conv_output_length(len_dim3, self.pool_size[2],
                                                    self.padding, self.strides[2])
        if self.data_format == 'channels_first':
            return (input_shape[0],
                    input_shape[1],
                    len_dim1, len_dim2, len_dim3)
        elif self.data_format == 'channels_last':
            return (input_shape[0],
                    len_dim1, len_dim2, len_dim3,
                    input_shape[4])

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(_ComplexPooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ComplexMaxPooling1D(_ComplexPooling1D):
    """ComplexMaxPooling1D"""
    def __init__(self, pool_size=(2,), strides=(1,), padding='same', data_format='channels_last', **kwargs):
        super(ComplexMaxPooling1D, self).__init__(pool_size, strides, padding,
                                                  data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        input_real, input_imag = complex_to_real_imag(inputs)
        real_outputs = KL.MaxPooling1D(pool_size, strides, padding)(input_real)
        imag_outputs = KL.MaxPooling1D(pool_size, strides, padding)(input_imag)
        outputs = real_imag_to_complex(real_outputs, imag_outputs)
        return outputs

class ComplexMaxPooling2D(_ComplexPooling2D):
    """ComplexMaxPooling2D"""
    def __init__(self, pool_size=(2, 2), strides=(1, 1), padding='same', data_format='channels_last', **kwargs):
        super(ComplexMaxPooling2D, self).__init__(pool_size, strides, padding,
                                                  data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        input_real, input_imag = complex_to_real_imag(inputs)
        real_outputs = KL.MaxPooling2D(pool_size, strides, padding)(input_real)
        imag_outputs = KL.MaxPooling2D(pool_size, strides, padding)(input_imag)
        outputs = real_imag_to_complex(real_outputs, imag_outputs)
        return outputs

class ComplexMaxPooling3D(_ComplexPooling3D):
    """ComplexMaxPooling3D"""
    def __init__(self, pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same', data_format='channels_last', **kwargs):
        super(ComplexMaxPooling3D, self).__init__(pool_size, strides, padding,
                                                  data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        input_real, input_imag = complex_to_real_imag(inputs)
        real_outputs = KL.MaxPooling3D(pool_size, strides, padding)(input_real)
        imag_outputs = KL.MaxPooling3D(pool_size, strides, padding)(input_imag)
        outputs = real_imag_to_complex(real_outputs, imag_outputs)
        return outputs

class ComplexAveragePooling1D(_ComplexPooling1D):
    """ComplexAveragePooling1D"""
    def __init__(self, pool_size=(2,), strides=(1,), padding='same', data_format='channels_last', **kwargs):
        super(ComplexAveragePooling1D, self).__init__(pool_size, strides, padding,
                                                      data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        input_real, input_imag = complex_to_real_imag(inputs)
        real_outputs = KL.AveragePooling1D(pool_size, strides, padding)(input_real)
        imag_outputs = KL.AveragePooling1D(pool_size, strides, padding)(input_imag)
        outputs = real_imag_to_complex(real_outputs, imag_outputs)
        return outputs

class ComplexAveragePooling2D(_ComplexPooling2D):
    """ComplexAveragePooling2D"""
    def __init__(self, pool_size=(2, 2), strides=(1, 1), padding='same', data_format='channels_last', **kwargs):
        super(ComplexAveragePooling2D, self).__init__(pool_size, strides, padding,
                                                      data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        input_real, input_imag = complex_to_real_imag(inputs)
        real_outputs = KL.AveragePooling2D(pool_size, strides, padding)(input_real)
        imag_outputs = KL.AveragePooling2D(pool_size, strides, padding)(input_imag)
        outputs = real_imag_to_complex(real_outputs, imag_outputs)
        return outputs

class ComplexAveragePooling3D(_ComplexPooling3D):
    """ComplexAveragePooling3D"""
    def __init__(self, pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same', data_format='channels_last', **kwargs):
        super(ComplexAveragePooling3D, self).__init__(pool_size, strides, padding,
                                                      data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        input_real, input_imag = complex_to_real_imag(inputs)
        real_outputs = KL.AveragePooling3D(pool_size, strides, padding)(input_real)
        imag_outputs = KL.AveragePooling3D(pool_size, strides, padding)(input_imag)
        outputs = real_imag_to_complex(real_outputs, imag_outputs)
        return outputs

if __name__ == "__main__":
    import __main__ as SP
#    inputs = KL.Input(shape=(128, 128, 2))
#    x = SP.ComplexMaxPooling2D()(inputs)
#    print(x)
#    print('x.int_shape', KB.int_shape(x))
#
#    inputs = KL.Input(shape=(128, 128, 2))
#    x = SP.ComplexMaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
#    print(x)
#    print('x.int_shape', KB.int_shape(x))


    inputs = KL.Input(shape=(128, 128, 2))
    x = SP.SpectralPooling2D(gamma=[0.15, 0.15])(inputs)
    print(x)
    print('x.int_shape', KB.int_shape(x))
