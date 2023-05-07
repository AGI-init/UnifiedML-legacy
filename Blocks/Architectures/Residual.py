# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

from omegaconf import OmegaConf

import Utils


class Residual(torch.nn.Module):
    """
    Residual with support for command-line instantiation and down-sampling
    """
    def __init__(self, model, down_sample=None, mode=torch.add, **kwargs):
        super().__init__()

        self.mode = mode  # Additive residual by default

        # Can pass a model in as an argument or via the command-line syntax
        self.model = Utils.instantiate(OmegaConf.create({'_target_': model}), **kwargs) if isinstance(model, str) \
            else model

        if 'input_shape' in kwargs:
            kwargs['output_shape'] = self.repr_shape(*kwargs['input_shape'])

        # Can pass a down-sampling model in as an argument or via the command-line syntax
        self.down_sample = Utils.instantiate(OmegaConf.create({'_target_': down_sample}),
                                             **kwargs) if isinstance(down_sample, str) \
            else down_sample

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.model)  # Note: if model/down-sample shapes mismatch, might return wrong

    def forward(self, input):
        output = self.model(input)

        if self.down_sample is not None:
            input = self.down_sample(input)

        return self.mode(output, input)  # Residual
