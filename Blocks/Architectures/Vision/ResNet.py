# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

from Blocks.Architectures.Residual import Residual
from Blocks.Architectures.Vision.CNN import AvgPool

import Utils


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, down_sample=None):
        super().__init__()

        if down_sample is None and (in_channels != out_channels or stride != 1):
            down_sample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                  kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(out_channels))

        pre_residual = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                               kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=kernel_size, padding=kernel_size // 2,
                                               bias=False),
                                     nn.BatchNorm2d(out_channels))

        self.Residual_block = nn.Sequential(Residual(pre_residual, down_sample),
                                            nn.ReLU(inplace=True))

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape([c, h, w], self.Residual_block)

    def forward(self, x):
        return self.Residual_block(x)


class MiniResNet(nn.Module):
    def __init__(self, input_shape, kernel_size=3, stride=2, dims=None, depths=None, output_dim=None):
        super().__init__()

        self.input_shape = input_shape
        in_channels = input_shape[0]

        if dims is None:
            dims = [32, 32]  # MiniResNet
        self.dims = dims

        if depths is None:
            depths = [3]  # MiniResNet
        self.depths = depths

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.trunk = nn.Sequential(nn.Conv2d(in_channels, dims[0],
                                             kernel_size=kernel_size, padding=1, bias=False),
                                             # kernel_size=7, stride=2, padding=3, bias=False),  # Pytorch settings
                                   nn.BatchNorm2d(dims[0]),
                                   nn.ReLU(inplace=True),
                                   # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                   )

        # CNN ResNet
        self.ResNet = nn.Sequential(*[nn.Sequential(*[ResidualBlock(dims[i + (j > 0)], dims[i + 1], kernel_size,
                                                                    1 + (stride - 1) * (i > 0 and j > 0))
                                                      for j in range(depth)])
                                      for i, depth in enumerate(depths)])

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(AvgPool(), nn.Linear(dims[-1], output_dim))

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape([c, h, w], self.trunk, self.ResNet, self.project)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        x = torch.cat(
            [context.view(*context.shape[:-3], -1, *self.input_shape[1:]) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *self.input_shape[1:]) if context.shape[-1]
                                                                                  % math.prod(self.input_shape[1:]) == 0
            else context.view(*context.shape, 1, 1).expand(*context.shape, *self.input_shape[1:])
             for context in x if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = x.shape[:-3]
        # Operate on last 3 dims
        x = x.view(-1, *x.shape[-3:])

        x = self.trunk(x)
        x = self.ResNet(x)
        x = self.project(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


class ResNet18(MiniResNet):
    def __init__(self, input_shape, output_dim=None):
        super().__init__(input_shape, 3, 2, [64, 64, 128, 256, 512], [2, 2, 2, 2], output_dim)


class ResNet50(MiniResNet):
    def __init__(self, input_shape, output_dim=None):
        super().__init__(input_shape, 3, 2, [64, 64, 128, 256, 512], [3, 4, 6, 3], output_dim)
