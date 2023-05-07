# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils

from Blocks.Augmentations import RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsemblePiActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning


class DQNAgent(torch.nn.Module):
    """Deep Q-Learning
    Generalized to continuous action spaces and classify tasks"""
    def __init__(self,
                 obs_spec, action_spec, num_actions, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 rand_steps, stddev_schedule,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 ):
        super().__init__()

        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1

        # Discrete RL
        assert discrete and RL, f'{type(self).__name__} only supports discrete RL. Set "discrete=true RL=true".'
        assert not generate, f'{type(self).__name__} does not support generative modeling.'

        # Continuous -> discrete conversion
        if not action_spec.discrete:
            assert num_actions > 1, 'Num actions cannot be 1 when converting action spaces; try the ' \
                                    '"num_actions=" flag (>1) to divide each action dimension into discrete bins.'

            action_spec.discrete_bins = num_actions  # Continuous env has no discrete bins by default, must specify

        # Image augmentation
        self.aug = Utils.instantiate(recipes.aug) or RandomShiftsAug(pad=4)

        self.encoder = CNNEncoder(obs_spec, standardize=standardize, **recipes.encoder, parallel=parallel, lr=lr)

        self.actor = EnsemblePiActor(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.actor,
                                     discrete=True, stddev_schedule=stddev_schedule, rand_steps=rand_steps)

        # Since discrete, Critic <- Actor
        recipes.critic.trunk = self.actor.trunk
        recipes.critic.Q_head = self.actor.Pi_head.ensemble

        self.critic = EnsembleQCritic(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.critic,
                                      discrete=True, lr=lr, ema_decay=ema_decay)

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).float()

            obs = self.encoder(obs)

            Pi = self.actor(obs, self.step)
            action = Pi.sample() if self.training else Pi.best

            if self.training:

                self.step += 1
                self.frame += len(obs)

            return action, {}

    def learn(self, replay):

        # Online RL
        assert not replay.offline, f'{type(self).__name__} does not support offline learning. ' \
                                   'Set "offline=false" or "online=true".'

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, *_ = Utils.to_torch(
            batch, self.device)

        instruct = label.nelement()  # Are labels present?

        # Supervised -> RL conversion
        if instruct:
            reward = (action.squeeze(1) == label).float()  # reward = correct, Note: Classify Env already sets this

        logs = {'time': time.time() - self.birthday, 'step': self.step, 'frame': self.frame,
                'episode': self.episode} if self.log else None

        # Augment, encode present
        obs = self.aug(obs)
        obs = self.encoder(obs)

        if replay.nstep:
            with torch.no_grad():
                # Augment, encode future
                next_obs = self.aug(next_obs)
                next_obs = self.encoder(next_obs)

        # Critic loss
        critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                  obs, action, reward, discount, next_obs,
                                                  self.step, logs=logs)

        # Update encoder and critic
        Utils.optimize(critic_loss,
                       self.encoder, self.critic)

        return logs
