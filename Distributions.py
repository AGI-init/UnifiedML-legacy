# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch.distributions import Normal, Categorical
from torch.distributions.utils import _standard_normal

import Utils


class TruncatedNormal(Normal):
    """
    A Gaussian Normal distribution generalized to multi-action sampling and the option to clip standard deviation.
    """
    def __init__(self, loc, scale, low=None, high=None, eps=1e-6, stddev_clip=None):
        super().__init__(loc, scale)

        # Clip range of samples
        self.low, self.high = low, high  # Ranges
        self.eps = eps  # Fringes

        # Clip range of standard deviation
        self.stddev_clip = stddev_clip  # -low, high

    def log_prob(self, value):
        if value.shape[-len(self.loc.shape):] == self.loc.shape:
            return super().log_prob(value)  # Inherit log_prob(•)
        else:
            # To account for batch_first=True
            b, *shape = self.loc.shape  # Assumes a single batch dim
            return super().log_prob(value.view(b, -1, *shape).transpose(0, 1)).transpose(0, 1).view(value.shape)

    def sample(self, sample_shape=1, to_clip=False, batch_first=True, keepdim=True):
        with torch.no_grad():
            return self.rsample(sample_shape, to_clip, batch_first, keepdim)

    def rsample(self, sample_shape=1, to_clip=True, batch_first=True, keepdim=True):
        if isinstance(sample_shape, int):
            sample_shape = torch.Size((sample_shape,))

        # Draw multiple samples
        shape = self._extended_shape(sample_shape)

        rand = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)  # Explore
        dev = rand * self.scale.expand(shape)  # Deviate

        if to_clip:
            dev = Utils.rclamp(dev, -self.stddev_clip, self.stddev_clip)  # Don't explore /too/ much, clip std
        x = self.loc.expand(shape) + dev

        if batch_first:
            x = x.transpose(0, len(sample_shape))  # Batch dim first, assumes single batch dim

            if keepdim:
                x = x.flatten(1, len(sample_shape) + 1)

        if self.low is not None and self.high is not None:
            # Differentiable truncation
            return Utils.rclamp(x, self.low + self.eps, self.high - self.eps)  # Clip sample

        return x


class NormalizedCategorical(Categorical):
    """
    A Categorical that normalizes samples, allows sampling along specific "dim"s, and can temperature-weigh the softmax.
    """
    def __init__(self, logits, low=None, high=None, temp=torch.ones(()), dim=-1):
        super().__init__(logits=logits.movedim(dim, -1))

        temp = torch.as_tensor(temp, device=logits.device, dtype=logits.dtype).expand_as(logits).movedim(dim, -1)

        self.logits /= temp

        self.low, self.high = (None, None) if low == 0 and high == logits.shape[-1] else (low, high)
        self.dim = dim

    def log_prob(self, value=None):
        if value is None:
            return self.logits.movedim(self.dim, -1)
        elif value.shape[-self.logits.dim():] == self.logits.shape:
            return super().log_prob(self.un_normalize(value))  # Un-normalized log_prob(•)
        else:
            # To account for batch_first=True
            b, *shape = self.logits.shape  # Assumes a single batch dim
            return super().log_prob(self.un_normalize(value.view(b, -1, *shape[:-1]).transpose(0, 1))).transpose(0, 1)

    def sample(self, sample_shape=1, batch_first=True):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        sample = super().sample(sample_shape)

        if batch_first:
            sample = sample.transpose(0, len(sample_shape))  # Batch dim first

        return self.normalize(sample)

    def rsample(self, *args, **kwargs):
        return self.sample(*args, **kwargs)  # Non-differentiable

    def normalize(self, sample):
        # Normalize -> [low, high]
        return sample / (self.logits.shape[-1] - 1) * (self.high - self.low) + self.low if self.low or self.high \
            else sample

    def un_normalize(self, value):
        # Inverse of normalize -> indices
        return (value - self.low) / (self.high - self.low) * (self.logits.shape[-1] - 1) if self.low or self.high \
            else value
