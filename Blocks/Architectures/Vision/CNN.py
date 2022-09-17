import math

import torch
from torch import nn

import Utils


class CNN(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, batch_norm=False, last_relu=True,
                 kernel_size=3, stride=2, padding=0, output_dim=None):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = [input_shape]

        in_channels, *self.spatial_shape = input_shape

        self.CNN = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else out_channels,
                                      out_channels, kernel_size, stride=stride if i == 0 else 1,
                                      padding=padding),
                            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                            nn.ReLU() if i < depth or last_relu else nn.Identity()) for i in range(depth + 1)],
        )

        if output_dim is not None:
            c, h, w = Utils.cnn_feature_shape(input_shape, self.CNN)
            feature_shape = c * h * w

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.Flatten(), nn.Linear(feature_shape, 50), nn.ReLU(), nn.Linear(50, output_dim))

        self.apply(Utils.weight_init)

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape([c, h, w], self.CNN, self.project)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        hw = self.spatial_shape if len(self.spatial_shape) == 2 else x[0].shape[-2:]
        x = torch.cat(
            [context.view(*context.shape[:-3], -1, *hw) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *hw) if context.shape[-1] % math.prod(hw) == 0
             else context.view(*context.shape, 1, 1).expand(*context.shape, *hw)
             for context in x if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = x.shape[:-3]
        # Operate on last 3 dims
        x = x.view(-1, *x.shape[-3:])

        x = self.CNN(x)
        x = self.project(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


class AvgPool(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def repr_shape(self, dim, *_):
        return dim,

    def forward(self, input):
        for _ in input.shape[2:]:
            input = input.mean(-1)
        return input
