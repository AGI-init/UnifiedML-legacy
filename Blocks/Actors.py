# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn

from Distributions import TruncatedNormal, NormalizedCategorical

from Blocks.Architectures.MLP import MLP

import Utils


class EnsemblePiActor(nn.Module):
    """Ensemble of Gaussian or Categorical policies Pi, generalized to discrete or continuous action spaces."""
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_spec, trunk=None, Pi_head=None, ensemble_size=2,
                 discrete=False, stddev_schedule=1, stddev_clip=torch.inf, optim=None, scheduler=None,
                 lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.num_actions = action_spec.discrete_bins or 1  # n
        self.action_dim = math.prod(action_spec.shape) * (1 if stddev_schedule else 2)  # d, or d * 2

        self.low, self.high = action_spec.low, action_spec.high

        # Standard dev value, max cutoff clip for action sampling
        self.stddev_schedule, self.stddev_clip = stddev_schedule, stddev_clip

        in_dim = math.prod(repr_shape)

        self.trunk = Utils.instantiate(trunk, input_shape=repr_shape, output_dim=trunk_dim) or nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.Tanh())

        in_shape = Utils.cnn_feature_shape(repr_shape, self.trunk)  # Will be trunk_dim when possible
        out_dim = self.num_actions * self.action_dim

        # Ensemble
        self.Pi_head = Utils.Ensemble([Utils.instantiate(Pi_head, i, input_shape=in_shape, output_dim=out_dim)
                                       or MLP(in_shape, out_dim, hidden_dim, 2) for i in range(ensemble_size)])

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).eval()

    def forward(self, obs, step=1):
        obs = self.trunk(obs)

        mean = self.Pi_head(obs).unflatten(-1, (self.num_actions, self.action_dim))  # [b, e, n, d or 2 * d]

        if self.stddev_schedule is None:
            mean, log_stddev = mean.chunk(2, dim=-1)  # [b, e, n, d]
            stddev = log_stddev.exp()  # [b, e, n, d]
        else:
            stddev = torch.full_like(mean, Utils.schedule(self.stddev_schedule, step))  # [b, e, n, d]

        if self.discrete:
            logits, ind = mean.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(stddev, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble st. dev [b, n, d]

            Pi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)

            # All actions' Q-values
            setattr(Pi, 'All_Qs', mean)  # [b, e, n, d]
        else:
            if self.low or self.high:
                mean = (torch.tanh(mean) + 1) / 2 * (self.high - self.low) + self.low  # Normalize  [b, e, n, d]

            Pi = TruncatedNormal(mean, stddev, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

        return Pi


class CategoricalCriticActor(nn.Module):  # a.k.a. "Creator"
    """Selects over actions based on Q-values."""
    def __init__(self, temp_schedule=1):
        super().__init__()

        self.temp_schedule = temp_schedule

    def forward(self, Qs, step=None, action=None):
        # Q-values per action
        q = Qs.mean(1)  # Mean-reduced ensemble

        # Normalize
        q -= q.max(-1, keepdim=True)[0]

        # Softmax temperature
        temp = Utils.schedule(self.temp_schedule, step) if step else 1

        # Categorical dist
        Psi = torch.distributions.Categorical(logits=q / temp)

        # Highest Q-value
        _, best_ind = q.max(-1)

        # Action corresponding to highest Q-value
        setattr(Psi, 'best', best_ind if action is None else Utils.gather(action, best_ind.unsqueeze(-1), 1).squeeze(1))

        # Action sampling
        sampler = Psi.sample
        Psi.sample = sampler if action is None else lambda: Utils.gather(action, sampler().unsqueeze(-1), 1).squeeze(1)

        return Psi
