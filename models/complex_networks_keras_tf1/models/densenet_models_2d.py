# -*- coding: utf-8 -*-

"""This module implements a number of popular two-dimensional complex valued DenseNet models."""

#  Authors: Qinggang Sun
#
#  Reference:
#       Somshubra Majumdar. DenseNet
#       https://github.com/titu1994/DenseNet

# pylint: disable=too-many-locals, dangerous-default-value, keyword-arg-before-vararg, too-many-arguments, invalid-name
# pylint: disable=too-many-branches, too-many-statements

import keras.backend as K
import keras.layers
import keras.regularizers

from .densenet_blocks_2d import dense2d_block, transition2d_block
from ..layers.activations import layer_activation
from ..layers.bn import ComplexBatchNormalization
from ..layers.conv import ComplexConv2D
from ..layers.dense import ComplexDense
# from ..layers.conv import conv2d_transpose
from ..layers.pool import ComplexMaxPooling2D, ComplexAveragePooling2D, SpectralPooling2D


class DenseNet2D(keras.Model):
    """Instantiate the DenseNet architecture, constructs a `keras.models.Model` object using the given block count.

        Args:
            inputs (keras tensor): e.g. an instance of `keras.layers.Input`
            depth (int, optional): number of layers in the DenseNet. Defaults to None.
            nb_dense_block (int, optional): number of dense blocks to add to end. Defaults to 4.
            growth_rate (int, optional): number of filters to add per dense block. Defaults to 12.
            nb_filter (int, optional): initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate. Defaults to -1.
            nb_layers_per_block (list, optional): number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1). Defaults to -1.
            bottleneck (bool, optional): flag to add bottleneck blocks in between dense blocks. Defaults to False.
            reduction (float, optional): reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression. Defaults to 0.0.
            dropout_rate (float, optional): dropout rate. Defaults to 0.0.
            weight_decay (float, optional): weight decay rate. Defaults to 1e-4.
            subsample_initial_block (bool, optional): Set to True to subsample the initial convolution and
                add a Pool2D layer before the dense blocks are added. Defaults to False.
            activation_conv (str, optional): Type of activation of conv layers in the blocks. Defaults to 'crelu'.
            pooling_func (list, optional): Type of pooling layers. Defaults to ['max', 'global_average'].
            include_top (bool, optional): whether to include the fully-connected
                layer at the top of the network. Defaults to False.
            classes (int, optional): number of classes to classify, only to be specified if `include_top` is True.
                Defaults to None.
            output_activation (str, optional): Type of activation at the top layer. Defaults to None.
    """

    def __init__(self,
                 inputs,
                 depth=None,
                 nb_dense_block=4,
                 growth_rate=12,
                 nb_filter=-1,
                 nb_layers_per_block=-1,
                 bottleneck=False,
                 reduction=0.0,
                 dropout_rate=0.0,
                 weight_decay=1e-4,
                 subsample_initial_block=False,
                 activation_conv='crelu',
                 pooling_func=['max', 'global_average'],
                 include_top=False,
                 classes=None,
                 output_activation=None,
                 *args,
                 **kwargs
                ):
        concat_axis = -1 if K.image_data_format() == "channels_last" else 1

        if reduction != 0.0:
            assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

        # layers in each dense block
        if isinstance(nb_layers_per_block, (list, tuple)):
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. ' \
                'Note that list size must be (nb_dense_block)'
            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (7, 7)
            initial_strides = (2, 2)
        else:
            initial_kernel = (3, 3)
            initial_strides = (1, 1)

        x_complex = ComplexConv2D(nb_filter, initial_kernel, strides=initial_strides,
                                  padding='same', use_bias=False,
                                  spectral_parametrization=False,
                                  kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)

        if subsample_initial_block:
            x_complex = ComplexBatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x_complex)
            x_complex = layer_activation(x_complex, activation_conv)
            if pooling_func[0] == 'max':
                x_complex = ComplexMaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_complex)
            elif pooling_func[0] == 'average':
                x_complex = ComplexAveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x_complex)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x_complex, nb_filter = dense2d_block(x_complex, nb_layers[block_idx], nb_filter, growth_rate,
                                                 activation=activation_conv, bottleneck=bottleneck,
                                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

            # add transition_block
            x_complex = transition2d_block(x_complex, nb_filter, activation=activation_conv,
                                           compression=compression, weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)

        # The last dense_block does not have a transition_block
        x_complex, nb_filter = dense2d_block(x_complex, final_nb_layer, nb_filter, growth_rate,
                                             activation=activation_conv, bottleneck=bottleneck,
                                             dropout_rate=dropout_rate, weight_decay=weight_decay)

        x_complex = ComplexBatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x_complex)
        x_complex = layer_activation(x_complex, activation_conv)

        if include_top:
            assert classes > 0
            if pooling_func[1] == 'global_average':
                x_complex = keras.layers.GlobalAveragePooling2D(name="pool5")(x_complex)
            elif pooling_func[1] == 'complex_average':
                x_complex = ComplexAveragePooling2D(name='pool5')(x_complex)
            elif pooling_func[1] == 'complex_max':
                x_complex = ComplexMaxPooling2D(name='pool5')(x_complex)
            elif pooling_func[1] == 'spectral_average':
                x_complex = SpectralPooling2D(gamma=[0.25, 0.25], name='pool5')(x_complex)

            if output_activation is None:
                output_activation = 'softmax'

            if K.ndim(x_complex) > 2:
                x_complex = keras.layers.Flatten()(x_complex)

            if output_activation.startswith('complex_'):
                output_activation = output_activation[len('complex_'):]
                x = ComplexDense(classes, activation=output_activation)(x_complex)
            else:
                x = keras.layers.Dense(classes, activation=output_activation)(x_complex)
        else:
            x = x_complex

        super(DenseNet2D, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)


class DenseNet2D121(DenseNet2D):
    """Constructs a `keras.models.Model` according to the DenseNet2D121 specifications.

    Args:
        inputs (keras tensor): e.g. an instance of `keras.layers.Input`.
        depth (int, optional): number of layers in the DenseNet. Defaults to 121.
        nb_dense_block (int, optional): number of dense blocks to add to end. Defaults to 4.
        growth_rate (int, optional): number of filters to add per dense block. Defaults to 32.
        nb_filter (int, optional): initial number of filters. -1 indicates initial
            number of filters is 2 * growth_rate. Defaults to 64.
        nb_layers_per_block (list, optional): number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1). Defaults to [6, 12, 24, 16].
        bottleneck (bool, optional): flag to add bottleneck blocks in between dense blocks. Defaults to True.
        reduction (float, optional): reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression. Defaults to 0.5.
        dropout_rate (float, optional): dropout rate. Defaults to 0.0.
        weight_decay (float, optional): weight decay rate. Defaults to 1e-4.
        subsample_initial_block (bool, optional): Set to True to subsample the initial convolution and
            add a Pool2D layer before the dense blocks are added. Defaults to True.
        activation_conv (str, optional): Type of activation of conv layers in the blocks. Defaults to 'crelu'.
        pooling_func (list, optional): Type of pooling layers. Defaults to ['max', 'global_average'].
        include_top (bool, optional): whether to include the fully-connected
            layer at the top of the network. Defaults to True.
        classes (int, optional): number of classes to classify, only to be specified if `include_top` is True.
            Defaults to 10.
        output_activation (str, optional): Type of activation at the top layer. Defaults to 'softmax'.
    Usage:
        >>> from complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D121
        >>> from keras.layers import Input
        >>> id1, id2, od = 128, 128, 3
        >>> inputs = Input(shape=(id1, id2, 2))
        >>> model_densenet = DenseNet2D121(
                inputs,
                depthh=121,
                nb_dense_block=4,
                growth_rate=13,
                nb_filter=64,
                nb_layers_per_block=[6, 12, 24, 16],
                bottleneck=True,
                reduction=0.5,
                dropout_rate=0.0,
                weight_decay=1e-4,
                subsample_initial_block=True,
                classes=od,
                output_activation='sigmoid')
        >>> print(model_densenet.summary())
    """

    def __init__(self,
                 inputs,
                 depth=121,
                 nb_dense_block=4,
                 growth_rate=32,
                 nb_filter=64,
                 nb_layers_per_block=[6, 12, 24, 16],
                 bottleneck=True,
                 reduction=0.5,
                 dropout_rate=0.0,
                 weight_decay=1e-4,
                 subsample_initial_block=True,
                 activation_conv='crelu',
                 pooling_func=['max', 'global_average'],
                 include_top=True,
                 classes=10,
                 output_activation='softmax',
                 *args,
                 **kwargs
                ):
        super(DenseNet2D121, self).__init__(
            inputs,
            depth,
            nb_dense_block,
            growth_rate,
            nb_filter,
            nb_layers_per_block,
            bottleneck,
            reduction,
            dropout_rate,
            weight_decay,
            subsample_initial_block,
            activation_conv,
            pooling_func,
            include_top,
            classes,
            output_activation,
            *args,
            **kwargs
        )


class DenseNet2D169(DenseNet2D):
    """Constructs a `keras.models.Model` according to the DenseNet2D169 specifications.

    Args:
        inputs (keras tensor): e.g. an instance of `keras.layers.Input`.
        depth (int, optional): number of layers in the DenseNet. Defaults to 169.
        nb_dense_block (int, optional): number of dense blocks to add to end. Defaults to 4.
        growth_rate (int, optional): number of filters to add per dense block. Defaults to 32.
        nb_filter (int, optional): initial number of filters. -1 indicates initial
            number of filters is 2 * growth_rate. Defaults to 64.
        nb_layers_per_block (list, optional): number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1). Defaults to [6, 12, 24, 16].
        bottleneck (bool, optional): flag to add bottleneck blocks in between dense blocks. Defaults to True.
        reduction (float, optional): reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression. Defaults to 0.5.
        dropout_rate (float, optional): dropout rate. Defaults to 0.0.
        weight_decay (float, optional): weight decay rate. Defaults to 1e-4.
        subsample_initial_block (bool, optional): Set to True to subsample the initial convolution and
            add a Pool2D layer before the dense blocks are added. Defaults to True.
        activation_conv (str, optional): Type of activation of conv layers in the blocks. Defaults to 'crelu'.
        pooling_func (list, optional): Type of pooling layers. Defaults to ['max', 'global_average'].
        include_top (bool, optional): whether to include the fully-connected
            layer at the top of the network. Defaults to True.
        classes (int, optional): number of classes to classify, only to be specified if `include_top` is True.
            Defaults to 10.
        output_activation (str, optional): Type of activation at the top layer. Defaults to 'softmax'.
    Usage:
        >>> from complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D169
        >>> from keras.layers import Input
        >>> id1, id2, od = 128, 128, 3
        >>> inputs = Input(shape=(id1, id2, 2))
        >>> model_densenet = DenseNet2D169(
                inputs,
                depthh=169,
                nb_dense_block=4,
                growth_rate=13,
                nb_filter=64,
                nb_layers_per_block=[6, 12, 32, 32],
                bottleneck=True,
                reduction=0.5,
                dropout_rate=0.0,
                weight_decay=1e-4,
                subsample_initial_block=True,
                classes=od,
                output_activation='sigmoid')
        >>> print(model_densenet.summary())
    """

    def __init__(self,
                 inputs,
                 depth=169,
                 nb_dense_block=4,
                 growth_rate=32,
                 nb_filter=64,
                 nb_layers_per_block=[6, 12, 32, 32],
                 bottleneck=True,
                 reduction=0.5,
                 dropout_rate=0.0,
                 weight_decay=1e-4,
                 subsample_initial_block=True,
                 activation_conv='crelu',
                 pooling_func=['max', 'global_average'],
                 include_top=True,
                 classes=10,
                 output_activation='softmax',
                 *args,
                 **kwargs
                ):
        super(DenseNet2D169, self).__init__(
            inputs,
            depth,
            nb_dense_block,
            growth_rate,
            nb_filter,
            nb_layers_per_block,
            bottleneck,
            reduction,
            dropout_rate,
            weight_decay,
            subsample_initial_block,
            activation_conv,
            pooling_func,
            include_top,
            classes,
            output_activation,
            *args,
            **kwargs
        )


class DenseNet2D201(DenseNet2D):
    """Constructs a `keras.models.Model` according to the DenseNet2D201 specifications.

    Args:
        inputs (keras tensor): e.g. an instance of `keras.layers.Input`.
        depth (int, optional): number of layers in the DenseNet. Defaults to 201.
        nb_dense_block (int, optional): number of dense blocks to add to end. Defaults to 4.
        growth_rate (int, optional): number of filters to add per dense block. Defaults to 32.
        nb_filter (int, optional): initial number of filters. -1 indicates initial
            number of filters is 2 * growth_rate. Defaults to 64.
        nb_layers_per_block (list, optional): number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1). Defaults to [6, 12, 24, 16].
        bottleneck (bool, optional): flag to add bottleneck blocks in between dense blocks. Defaults to True.
        reduction (float, optional): reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression. Defaults to 0.5.
        dropout_rate (float, optional): dropout rate. Defaults to 0.0.
        weight_decay (float, optional): weight decay rate. Defaults to 1e-4.
        subsample_initial_block (bool, optional): Set to True to subsample the initial convolution and
            add a Pool2D layer before the dense blocks are added. Defaults to True.
        activation_conv (str, optional): Type of activation of conv layers in the blocks. Defaults to 'crelu'.
        pooling_func (list, optional): Type of pooling layers. Defaults to ['max', 'global_average'].
        include_top (bool, optional): whether to include the fully-connected
            layer at the top of the network. Defaults to True.
        classes (int, optional): number of classes to classify, only to be specified if `include_top` is True.
            Defaults to 10.
        output_activation (str, optional): Type of activation at the top layer. Defaults to 'softmax'.
    Usage:
        >>> from complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D201
        >>> from keras.layers import Input
        >>> id1, id2, od = 128, 128, 3
        >>> inputs = Input(shape=(id1, id2, 2))
        >>> model_densenet = DenseNet2D201(
                inputs,
                depthh=201,
                nb_dense_block=4,
                growth_rate=13,
                nb_filter=64,
                nb_layers_per_block=[6, 12, 48, 32],
                bottleneck=True,
                reduction=0.5,
                dropout_rate=0.0,
                weight_decay=1e-4,
                subsample_initial_block=True,
                classes=od,
                output_activation='sigmoid')
        >>> print(model_densenet.summary())
    """

    def __init__(self,
                 inputs,
                 depth=201,
                 nb_dense_block=4,
                 growth_rate=32,
                 nb_filter=64,
                 nb_layers_per_block=[6, 12, 48, 32],
                 bottleneck=True,
                 reduction=0.5,
                 dropout_rate=0.0,
                 weight_decay=1e-4,
                 subsample_initial_block=True,
                 activation_conv='crelu',
                 pooling_func=['max', 'global_average'],
                 include_top=True,
                 classes=10,
                 output_activation='softmax',
                 *args,
                 **kwargs
                ):
        super(DenseNet2D201, self).__init__(
            inputs,
            depth,
            nb_dense_block,
            growth_rate,
            nb_filter,
            nb_layers_per_block,
            bottleneck,
            reduction,
            dropout_rate,
            weight_decay,
            subsample_initial_block,
            activation_conv,
            pooling_func,
            include_top,
            classes,
            output_activation,
            *args,
            **kwargs
        )


class DenseNet2D264(DenseNet2D):
    """Constructs a `keras.models.Model` according to the DenseNet2D264 specifications.

    Args:
        inputs (keras tensor): e.g. an instance of `keras.layers.Input`.
        depth (int, optional): number of layers in the DenseNet. Defaults to 264.
        nb_dense_block (int, optional): number of dense blocks to add to end. Defaults to 4.
        growth_rate (int, optional): number of filters to add per dense block. Defaults to 32.
        nb_filter (int, optional): initial number of filters. -1 indicates initial
            number of filters is 2 * growth_rate. Defaults to 64.
        nb_layers_per_block (list, optional): number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1). Defaults to [6, 12, 24, 16].
        bottleneck (bool, optional): flag to add bottleneck blocks in between dense blocks. Defaults to True.
        reduction (float, optional): reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression. Defaults to 0.5.
        dropout_rate (float, optional): dropout rate. Defaults to 0.0.
        weight_decay (float, optional): weight decay rate. Defaults to 1e-4.
        subsample_initial_block (bool, optional): Set to True to subsample the initial convolution and
            add a Pool2D layer before the dense blocks are added. Defaults to True.
        activation_conv (str, optional): Type of activation of conv layers in the blocks. Defaults to 'crelu'.
        pooling_func (list, optional): Type of pooling layers. Defaults to ['max', 'global_average'].
        include_top (bool, optional): whether to include the fully-connected
            layer at the top of the network. Defaults to True.
        classes (int, optional): number of classes to classify, only to be specified if `include_top` is True.
            Defaults to 10.
        output_activation (str, optional): Type of activation at the top layer. Defaults to 'softmax'.
    Usage:
        >>> from complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D264
        >>> from keras.layers import Input
        >>> id1, id2, od = 128, 128, 3
        >>> inputs = Input(shape=(id1, id2, 2))
        >>> model_densenet = DenseNet2D264(
                inputs,
                depthh=264,
                nb_dense_block=4,
                growth_rate=13,
                nb_filter=64,
                nb_layers_per_block=[6, 12, 64, 48],
                bottleneck=True,
                reduction=0.5,
                dropout_rate=0.0,
                weight_decay=1e-4,
                subsample_initial_block=True,
                classes=od,
                output_activation='sigmoid')
        >>> print(model_densenet.summary())
    """

    def __init__(self,
                 inputs,
                 depth=264,
                 nb_dense_block=4,
                 growth_rate=32,
                 nb_filter=64,
                 nb_layers_per_block=[6, 12, 64, 48],
                 bottleneck=True,
                 reduction=0.5,
                 dropout_rate=0.0,
                 weight_decay=1e-4,
                 subsample_initial_block=True,
                 activation_conv='crelu',
                 pooling_func=['max', 'global_average'],
                 include_top=True,
                 classes=10,
                 output_activation='softmax',
                 *args,
                 **kwargs
                ):
        super(DenseNet2D264, self).__init__(
            inputs,
            depth,
            nb_dense_block,
            growth_rate,
            nb_filter,
            nb_layers_per_block,
            bottleneck,
            reduction,
            dropout_rate,
            weight_decay,
            subsample_initial_block,
            activation_conv,
            pooling_func,
            include_top,
            classes,
            output_activation,
            *args,
            **kwargs
        )


class DenseNet2D161(DenseNet2D):
    """Constructs a `keras.models.Model` according to the DenseNet2D161 specifications.

    Args:
        inputs (keras tensor): e.g. an instance of `keras.layers.Input`.
        depth (int, optional): number of layers in the DenseNet. Defaults to 161.
        nb_dense_block (int, optional): number of dense blocks to add to end. Defaults to 4.
        growth_rate (int, optional): number of filters to add per dense block. Defaults to 32.
        nb_filter (int, optional): initial number of filters. -1 indicates initial
            number of filters is 2 * growth_rate. Defaults to 64.
        nb_layers_per_block (list, optional): number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1). Defaults to [6, 12, 24, 16].
        bottleneck (bool, optional): flag to add bottleneck blocks in between dense blocks. Defaults to True.
        reduction (float, optional): reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression. Defaults to 0.5.
        dropout_rate (float, optional): dropout rate. Defaults to 0.0.
        weight_decay (float, optional): weight decay rate. Defaults to 1e-4.
        subsample_initial_block (bool, optional): Set to True to subsample the initial convolution and
            add a Pool2D layer before the dense blocks are added. Defaults to True.
        activation_conv (str, optional): Type of activation of conv layers in the blocks. Defaults to 'crelu'.
        pooling_func (list, optional): Type of pooling layers. Defaults to ['max', 'global_average'].
        include_top (bool, optional): whether to include the fully-connected
            layer at the top of the network. Defaults to True.
        classes (int, optional): number of classes to classify, only to be specified if `include_top` is True.
            Defaults to 10.
        output_activation (str, optional): Type of activation at the top layer. Defaults to 'softmax'.
    Usage:
        >>> from complex_networks_keras_tf1.models.densenet_models_2d import DenseNet2D161
        >>> from keras.layers import Input
        >>> id1, id2, od = 128, 128, 3
        >>> inputs = Input(shape=(id1, id2, 2))
        >>> model_densenet = DenseNet2D161(
                inputs,
                depthh=161,
                nb_dense_block=4,
                growth_rate=13,
                nb_filter=64,
                nb_layers_per_block=[6, 12, 36, 24],
                bottleneck=True,
                reduction=0.5,
                dropout_rate=0.0,
                weight_decay=1e-4,
                subsample_initial_block=True,
                classes=od,
                output_activation='sigmoid')
        >>> print(model_densenet.summary())
    """

    def __init__(self,
                 inputs,
                 depth=161,
                 nb_dense_block=4,
                 growth_rate=48,
                 nb_filter=96,
                 nb_layers_per_block=[6, 12, 36, 24],
                 bottleneck=True,
                 reduction=0.5,
                 dropout_rate=0.0,
                 weight_decay=1e-4,
                 subsample_initial_block=True,
                 activation_conv='crelu',
                 pooling_func=['max', 'global_average'],
                 include_top=True,
                 classes=10,
                 output_activation='softmax',
                 *args,
                 **kwargs
                ):
        super(DenseNet2D161, self).__init__(
            inputs,
            depth,
            nb_dense_block,
            growth_rate,
            nb_filter,
            nb_layers_per_block,
            bottleneck,
            reduction,
            dropout_rate,
            weight_decay,
            subsample_initial_block,
            activation_conv,
            pooling_func,
            include_top,
            classes,
            output_activation,
            *args,
            **kwargs
        )
