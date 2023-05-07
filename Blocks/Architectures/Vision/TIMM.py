# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn

import timm

from Blocks.Architectures.Vision.CNN import cnn_broadcast

import Utils


class TIMM(nn.Module):
    """Backwards compatibility with the TIMM (Pytorch Image Models) ecosystem. Download or load a model as follows.
    Usage:  python Run.py  task=classify/mnist  Eyes=Blocks.Architectures.Vision.TIMM.TIMM  +eyes.name=mobilenetv2_100
    Not installed by default. $ pip install timm  (that dollar sign is totally a Freudian slip)
    Models listed here:  https://rwightman.github.io/pytorch-image-models/models/
    """
    def __init__(self, input_shape, name, pretrained=False, detach=False, pool='avg', output_shape=None):
        super().__init__()

        self.input_shape, output_dim = Utils.to_tuple(input_shape), Utils.prod(output_shape)

        in_channels = self.input_shape[0]

        assert name in timm.list_models(pretrained=True), f'Could not find model {name} in TIMM models. ' \
                                                          f'Find a list of available models in the TIMM docs here: ' \
                                                          f'https://rwightman.github.io/pytorch-image-models/models/'

        self.model = timm.create_model(name, pretrained=pretrained, in_chans=in_channels,
                                       num_classes=0 if output_shape is None else output_dim,
                                       global_pool='' if output_shape is None else pool)

        self.detach = detach  # Fix weights

    def repr_shape(self, *_):
        return Utils.repr_shape(_, self.model)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        if self.detach:
            with torch.no_grad():
                x = self.model.eval()(x)  # Detach gradients from model
        else:
            x = self.model(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
