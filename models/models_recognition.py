# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:09:11 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling2D, Input, normalization, Flatten, Reshape, Dropout
from keras import regularizers
import logging

# build resnet_1d_50 from keras_resnet
import keras_resnet
import keras_resnet.models
def build_model5(id1, id2, od, **kwargs):  # ResNet 1D 50
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2))
    model_resnet = keras_resnet.models.ResNet0D50(
        input_frames, include_top=True, classes=od, freeze_bn=False, output_activation=output_activation)
    logging.info(model_resnet.summary())
    return model_resnet

# build resnet_1d_34 from keras_resnet
def build_model6(id1, id2, od, **kwargs):  # ResNet 1D 34
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2))
    model_resnet = keras_resnet.models.ResNet0D34(
        input_frames, include_top=True, classes=od, freeze_bn=False, output_activation=output_activation)
    logging.info(model_resnet.summary())
    return model_resnet

from .densenet_ankitvgupta.models.one_d_changed import DenseNet121
def build_model7(id1, id2, od, **kwargs):  # DenseNet 1D 121
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs.keys() else 1e-4
    input_frames = Input(shape=(id1, id2))
    model_densenet = DenseNet121(k=13, conv_kernel_width=3, bottleneck_size=52,
                                 transition_pool_size=2, transition_pool_stride=2, theta=0.5,
                                 initial_conv_width=7, initial_stride=2, initial_filters=64,
                                 initial_pool_width=3, initial_pool_stride=2, use_global_pooling=True,
                                 weight_decay=weight_decay)
    x = model_densenet(input_frames)
    x = Reshape((1, -1))(x)
    target = Dense(od, activation=output_activation)(x)
    model = Model(input_frames, [target])
    logging.info(model.summary())
    return model

from .densenet_ankitvgupta.models.one_d_changed import DenseNet169
def build_model8(id1, id2, od, **kwargs):  # DenseNet 1D 169
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs.keys() else 1e-4
    input_frames = Input(shape=(id1, id2))
    model_densenet = DenseNet169(k=10, conv_kernel_width=3, bottleneck_size=40,
                                 transition_pool_size=2, transition_pool_stride=2, theta=0.5,
                                 initial_conv_width=7, initial_stride=2, initial_filters=64,
                                 initial_pool_width=3, initial_pool_stride=2, use_global_pooling=True,
                                 weight_decay=weight_decay)
    x = model_densenet(input_frames)
    x = Reshape((1, -1))(x)
    target = Dense(od, activation=output_activation)(x)
    model = Model(input_frames, [target])
    logging.info(model.summary())
    return model

# build resnet_1d_10 from keras_resnet
def build_model90(id1, id2, od, **kwargs):  # ResNet 1D 10
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2))
    model_resnet = keras_resnet.models.ResNet0D18(input_frames,
                                                  blocks=[2, 2],
                                                  include_top=True, classes=od, freeze_bn=False,
                                                  output_activation=output_activation)
    logging.info(model_resnet.summary())
    return model_resnet

# build resnet_1d_18 from keras_resnet
def build_model9(id1, id2, od, **kwargs):  # ResNet 1D 18
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2))
    model_resnet = keras_resnet.models.ResNet0D18(
        input_frames, include_top=True, classes=od, freeze_bn=False, output_activation=output_activation)
    logging.info(model_resnet.summary())
    return model_resnet

def build_model10(id1, id2, od, **kwargs):  # ResNet 2D 50
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2, 1))
    model_resnet = keras_resnet.models.ResNet2D50(
        input_frames, include_top=True, classes=od, freeze_bn=False, output_activation=output_activation)
    logging.info(model_resnet.summary())
    return model_resnet

from .densenet_titu1994.densenet import DenseNet
def build_model11(id1, id2, od, **kwargs):  # DenseNet 2D 121
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs.keys() else 1e-4
    input_frames = Input(shape=(id1, id2, 1))
    model_densenet = DenseNet(input_shape=(id1, id2, 1),
                              depth=121, nb_dense_block=4, growth_rate=13, nb_filter=64,
                              nb_layers_per_block=[6, 12, 24, 16], bottleneck=True, reduction=0.5,
                              dropout_rate=0.0, weight_decay=weight_decay, subsample_initial_block=True,
                              include_top=False, weights=None, input_tensor=input_frames)
    x = model_densenet(input_frames)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, -1))(x)
    target = Dense(od, activation=output_activation)(x)
    model = Model(input_frames, [target])
    logging.info(model.summary())
    return model

def build_model12(id1, id2, od, **kwargs):  # ResNet 2D 18
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2, 1))
    model_resnet = keras_resnet.models.ResNet2D18(
        input_frames, include_top=True, classes=od, freeze_bn=False, output_activation=output_activation)
    logging.debug(f'model.inputs {model_resnet.inputs}')
    logging.info(model_resnet.summary())
    return model_resnet

def build_model13(id1, id2, od, **kwargs):  # ResNet 2D 34
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2, 1))
    model_resnet = keras_resnet.models.ResNet2D34(
        input_frames, include_top=True, classes=od, freeze_bn=False, output_activation=output_activation)
    logging.info(model_resnet.summary())
    return model_resnet

def build_model14(id1, id2, od, **kwargs):  # DenseNet 2D 169
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs.keys() else 1e-4
    input_frames = Input(shape=(id1, id2, 1))
    model_densenet = DenseNet(input_shape=(id1, id2, 1),
                              depth=169, nb_dense_block=4, growth_rate=10, nb_filter=64,
                              nb_layers_per_block=[6, 12, 32, 32], bottleneck=True, reduction=0.5,
                              dropout_rate=0.0, weight_decay=weight_decay, subsample_initial_block=True,
                              include_top=False, weights=None, input_tensor=input_frames)
    x = model_densenet(input_frames)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, -1))(x)
    target = Dense(od, activation=output_activation)(x)
    model = Model(input_frames, [target])
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.resnet_models_2d import ResNet2D18 as ComplexResNet2D18
def build_model15(id1, id2, od, **kwargs):  # Complex ResNet 2D 18
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2, 2))
    model = ComplexResNet2D18(input_frames, n_filters=32, include_top=True, classes=od,
                              pooling_func=['max', 'global_average'],
                              output_activation=output_activation)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.resnet_models_2d import ResNet2D34 as ComplexResNet2D34
def build_model16(id1, id2, od, **kwargs):  # Complex ResNet 2D 34
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2, 2))
    model = ComplexResNet2D34(input_frames, n_filters=32, include_top=True, classes=od,
                              pooling_func=['max', 'global_average'],
                              output_activation=output_activation)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.resnet_models_2d import ResNet2D50 as ComplexResNet2D50
def build_model17(id1, id2, od, **kwargs):  # Complex ResNet 2D 50
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2, 2))
    model = ComplexResNet2D50(input_frames, n_filters=32, include_top=True, classes=od,
                              pooling_func=['max', 'global_average'],
                              output_activation=output_activation)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D121 as ComplexDenseNet2D121
def build_model18(id1, id2, od, **kwargs):  # Complex DenseNet 2D 121
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2, 2))
    model = ComplexDenseNet2D121(input_frames,
                                 depth=121, nb_dense_block=4, growth_rate=13, nb_filter=32,
                                 nb_layers_per_block=[6, 12, 24, 16], reduction=0.5,
                                 dropout_rate=0.0, weight_decay=1e-4,
                                 pooling_func=['max', 'global_average'],
                                 subsample_initial_block=True,
                                 output_activation=output_activation, include_top=True, classes=od)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D169 as ComplexDenseNet2D169
def build_model19(id1, id2, od, **kwargs):  # Complex DenseNet 2D 169
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, id2, 2))
    model = ComplexDenseNet2D169(input_frames,
                                 depth=169, nb_dense_block=4, growth_rate=10, nb_filter=32,
                                 nb_layers_per_block=[6, 12, 32, 32], reduction=0.5,
                                 dropout_rate=0.0, weight_decay=1e-4,
                                 pooling_func=['max', 'global_average'],
                                 subsample_initial_block=True,
                                 output_activation=output_activation, include_top=True, classes=od)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.resnet_models_1d import ResNet1D18 as ComplexResNet1D18
def build_model20(id1, od, **kwargs):  # Complex ResNet 1D 18
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, 2))
    model = ComplexResNet1D18(input_frames, include_top=True, classes=od,
                              pooling_func=['max', 'global_average'],
                              output_activation=output_activation)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.resnet_models_1d import ResNet1D34 as ComplexResNet1D34
def build_model21(id1, od, **kwargs):  # Complex ResNet 1D 34
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, 2))
    model = ComplexResNet1D34(input_frames, include_top=True, classes=od,
                              pooling_func=['max', 'global_average'],
                              output_activation=output_activation)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.resnet_models_1d import ResNet1D50 as ComplexResNet1D50
def build_model22(id1, od, **kwargs):  # Complex ResNet 1D 50
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, 2))
    model = ComplexResNet1D50(input_frames, include_top=True, classes=od,
                              pooling_func=['max', 'global_average'],
                              output_activation=output_activation)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.densenet_models_1d import DenseNet1D121 as ComplexDenseNet1D121
def build_model23(id1, od, **kwargs):  # Complex DenseNet 1D 121
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, 2))
    model = ComplexDenseNet1D121(input_frames,
                                 depth=121, nb_dense_block=4, growth_rate=13, nb_filter=64,
                                 nb_layers_per_block=[6, 12, 24, 16], reduction=0.5,
                                 dropout_rate=0.0, weight_decay=1e-4,
                                 pooling_func=['max', 'global_average'],
                                 subsample_initial_block=True,
                                 output_activation=output_activation, include_top=True, classes=od)
    logging.info(model.summary())
    return model

from .complex_networks_keras_tf1.models.densenet_models_1d import DenseNet1D169 as ComplexDenseNet1D169
def build_model24(id1, od, **kwargs):  # Complex DenseNet 1D 169
    output_activation = kwargs['output_activation'] if 'output_activation' in kwargs.keys() else 'sigmoid'
    input_frames = Input(shape=(id1, 2))
    model = ComplexDenseNet1D169(input_frames,
                                 depth=169, nb_dense_block=4, growth_rate=10, nb_filter=64,
                                 nb_layers_per_block=[6, 12, 32, 32], reduction=0.5,
                                 dropout_rate=0.0, weight_decay=1e-4,
                                 pooling_func=['max', 'global_average'],
                                 subsample_initial_block=True,
                                 output_activation=output_activation, include_top=True, classes=od)
    logging.info(model.summary())
    return model