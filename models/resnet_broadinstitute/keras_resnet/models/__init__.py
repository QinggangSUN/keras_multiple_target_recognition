# -*- coding: utf-8 -*-

"""
keras_resnet.models
~~~~~~~~~~~~~~~~~~~

This module implements popular residual models.
"""

from ._0d import (
    ResNet0D,
    ResNet0D18,
    ResNet0D34,
    ResNet0D50,
    ResNet0D101,
    ResNet0D152,
    ResNet0D200
)

from ._1d import (
    ResNet1D,
    ResNet1D18,
    ResNet1D34,
    ResNet1D50,
    ResNet1D101,
    ResNet1D152,
    ResNet1D200
)

from ._2d import (
    ResNet2D,
    ResNet2D18,
    ResNet2D34,
    ResNet2D50,
    ResNet2D101,
    ResNet2D152,
    ResNet2D200
)

from ._3d import (
    ResNet3D,
    ResNet3D18,
    ResNet3D34,
    ResNet3D50,
    ResNet3D101,
    ResNet3D152,
    ResNet3D200
)

from ._feature_pyramid_2d import (
    FPN2D,
    FPN2D18,
    FPN2D34,
    FPN2D50,
    FPN2D101,
    FPN2D152,
    FPN2D200
)

from ._time_distributed_2d import (
    TimeDistributedResNet,
    TimeDistributedResNet18,
    TimeDistributedResNet34,
    TimeDistributedResNet50,
    TimeDistributedResNet101,
    TimeDistributedResNet152,
    TimeDistributedResNet200
)

# for backwards compatibility reasons
ResNet = ResNet2D
ResNet18 = ResNet2D18
ResNet34 = ResNet2D34
ResNet50 = ResNet2D50
ResNet101 = ResNet2D101
ResNet152 = ResNet2D152
ResNet200 = ResNet2D200
