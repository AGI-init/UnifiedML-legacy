# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import time

import torch

import Utils


class RandomAgent(torch.nn.Module):
    """Random Agent generalized to support everything"""
    def __init__(self,
                 obs_spec, action_spec, num_actions, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 rand_steps, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 ):
        super().__init__()

        self.device = device
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1

        action_dim = math.prod(obs_spec.shape) if generate else action_spec.shape[-1]

        self.actor = Utils.Rand(action_dim, uniform=True)

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.actor):
            obs = torch.as_tensor(obs, device=self.device).float()

            action = self.actor(obs) * 2 - 1  # [-1, 1]

            if self.training:
                self.step += 1
                self.frame += len(obs)

            return action, {}

    def learn(self, replay=None):
        return
