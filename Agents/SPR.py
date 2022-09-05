# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

from Blocks.Architectures import MLP
from Blocks.Architectures.Vision.ResNet import MiniResNet

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsemblePiActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class SPRAgent(torch.nn.Module):
    """Self-Predictive Representations (https://arxiv.org/abs/2007.05929)
    Generalized to continuous action spaces, classification, and generative modeling"""
    def __init__(self,
                 obs_spec, action_spec, num_actions, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 depth=5  # SPR
                 ):
        super().__init__()

        self.discrete = discrete and not generate  # Continuous supported!
        self.supervise = supervise  # And classification...
        self.RL = RL or generate
        self.generate = generate  # And generative modeling, too
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1
        self.explore_steps = explore_steps
        self.ema = ema

        self.num_actions = num_actions

        self.depth = depth  # Dynamics prediction depth

        # Image augmentation
        self.aug = Utils.instantiate(recipes.aug) or torch.nn.Sequential(IntensityAug(0.05), RandomShiftsAug(pad=4))

        # RL -> generate conversion
        if self.generate:
            standardize = False
            norm = True  # Normalize Obs to range [-1, 1]

            # Action = Imagined Obs
            action_spec.update({'shape': obs_spec.shape, 'discrete_bins': None,
                                'low': -1, 'high': 1, 'discrete': False})

            # Remove encoder, replace trunk with random noise
            recipes.encoder.Eyes = torch.nn.Identity()  # Generate "imagines" — no need for "seeing" with Eyes
            recipes.actor.trunk = Utils.Rand(size=trunk_dim)  # Generator observes random Gaussian noise as input

        # Discrete -> continuous conversion
        if action_spec.discrete and not self.discrete:
            # Normalizing actions to range [-1, 1] helps continuous RL
            action_spec.low, action_spec.high = (-1, 1) if self.RL else (None, None)

        # Continuous -> discrete conversion
        if self.discrete and not action_spec.discrete:
            assert self.num_actions > 1, 'Num actions cannot be 1 when discrete; try the "num_actions=" flag (>1) to ' \
                                         'divide each action dimension into discrete bins, or specify "discrete=false".'

            action_spec.discrete_bins = self.num_actions  # Continuous env has no discrete bins by default, must specify

        self.encoder = CNNEncoder(obs_spec, standardize=standardize, norm=norm, **recipes.encoder, parallel=parallel,
                                  lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay, ema_decay=ema_decay)

        self.actor = EnsemblePiActor(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.actor,
                                     ensemble_size=2 if self.discrete and self.RL else 1,
                                     discrete=self.discrete, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                     lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                     ema_decay=ema_decay * ema)

        # Dynamics
        if self.depth and not self.generate:
            shape = list(self.encoder.feature_shape)

            # Action -> One-Hot, if single-dim discrete, otherwise action shape
            self.action_dim = action_spec.discrete_bins if self.discrete and action_spec.shape == (1,) \
                else self.actor.num_actions * self.actor.action_dim if action_spec.discrete and not self.discrete \
                else self.actor.action_dim

            shape[0] += self.action_dim  # Predicting from obs and action

            resnet = MiniResNet(input_shape=shape, stride=1, dims=(64, self.encoder.feature_shape[0]), depths=(1,))

            self.dynamics = CNNEncoder(self.encoder.feature_shape, context_dim=self.action_dim,
                                       Eyes=torch.nn.Sequential(resnet, Utils.Norm(-3)),
                                       lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay)

            # Self supervisors
            self.projector = CNNEncoder(self.encoder.feature_shape,
                                        Eyes=MLP(self.encoder.feature_shape, hidden_dim, hidden_dim, 2),
                                        lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                        ema_decay=ema_decay)

            self.predictor = CNNEncoder(self.projector.repr_shape,
                                        Eyes=MLP(self.projector.repr_shape, hidden_dim, hidden_dim, 2),
                                        lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay)

        # When discrete, Critic <- Actor
        if self.discrete:
            recipes.critic.trunk = self.actor.trunk
            recipes.critic.Q_head = self.actor.Pi_head.ensemble

        self.critic = EnsembleQCritic(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.critic,
                                      ensemble_size=2 if self.RL else 1,
                                      discrete=self.discrete, ignore_obs=self.generate,
                                      lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                      ema_decay=ema_decay)

        self.action_selector = CategoricalCriticActor(stddev_schedule)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor, self.critic):
            obs = torch.as_tensor(obs, device=self.device).float()

            # Exponential moving average (EMA) shadows
            encoder = self.encoder.ema if self.ema else self.encoder
            actor = self.actor.ema if self.ema else self.actor
            critic = self.critic.ema if self.ema else self.critic

            # See
            obs = encoder(obs)

            # Act
            Pi = actor(obs, self.step)

            action = Pi.sample(self.num_actions) if self.training \
                else Pi.best if self.discrete \
                else Pi.mean

            if self.training:
                # Select among candidate actions based on Q-value
                if self.num_actions > 1:
                    All_Qs = getattr(Pi, 'All_Qs', None)  # Discrete Actor policy already knows all Q-values

                    action = self.action_selector(critic(obs, action, All_Qs), self.step, action).sample()

                self.step += 1
                self.frame += len(obs)

                if self.step < self.explore_steps and not self.generate:
                    # Explore
                    action.uniform_(actor.low or 1, actor.high or 9)  # Env will automatically round if discrete

            return action, {'step': self.step}

    # "Dream"
    def learn(self, replay):
        # "Recollect"

        batch = replay.sample(trajectories=True)
        obs, action, reward, discount, next_obs, label, *traj, step, ids, meta = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r, traj_l = traj

        # "Envision" / "Perceive"

        # Augment, encode present
        obs = self.aug(obs)
        features = self.encoder(obs, pool=False)
        obs = self.encoder.pool(features)

        if replay.nstep > 0 and not self.generate:
            with torch.no_grad():
                # Augment, encode future
                next_obs = self.aug(next_obs)
                next_obs = self.encoder(next_obs)

        # Actor-Critic -> Generator-Discriminator conversion
        if self.generate:
            action, reward[:] = obs, 1  # "Real"
            next_obs[:] = label[:] = float('nan')

        # "Journal teachings"

        offline = replay.offline

        logs = {'time': time.time() - self.birthday, 'step': self.step + offline, 'frame': self.frame + offline,
                'epoch' if offline else 'episode':  self.epoch if offline else self.episode} if self.log \
            else None

        if offline:
            self.step += 1
            self.frame += len(obs)
            self.epoch = replay.epoch

        instruction = ~torch.isnan(label)

        # "Acquire Wisdom"

        # Classification
        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # Inference
            Pi = self.actor(obs)

            y_predicted = (Pi.All_Qs if self.discrete else Pi.mean).mean(1)  # Average over ensembles

            mistake = cross_entropy(y_predicted, label.long(), reduction='none')
            correct = (y_predicted.argmax(1) == label).float()
            accuracy = correct.mean()

            if self.log:
                logs.update({'accuracy': accuracy})

            # Supervised learning
            if self.supervise:
                # Supervised loss
                supervised_loss = mistake.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.actor, epoch=self.epoch if offline else self.episode, retain_graph=True)

                if self.log:
                    logs.update({'supervised_loss': supervised_loss})

            # (Auxiliary) reinforcement
            if self.RL:
                half = len(obs) // 2
                mistake[:half] = cross_entropy(y_predicted[:half].uniform_(-1, 1),
                                               label[:half].long(), reduction='none')
                action = (y_predicted.argmax(1, keepdim=True) if self.discrete else y_predicted).detach()
                reward = -mistake.detach()  # reward = -error
                next_obs[:] = float('nan')

                if self.log:
                    logs.update({'reward': reward})

        # Reinforcement learning / generative modeling
        if self.RL:
            # "Imagine"

            # Generative modeling
            if self.generate:
                half = len(obs) // 2

                Pi = self.actor(obs[:half])
                generated_image = Pi.mean.flatten(1)

                action[:half], reward[:half] = generated_image, 0  # Discriminate "fake"

            # "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Dynamics loss
            dynamics_loss = 0 if replay.nstep == 0 or self.generate or not self.depth \
                else SelfSupervisedLearning.dynamicsLearning(features, traj_o, traj_a, traj_r,
                                                             self.encoder, self.dynamics, self.projector,
                                                             self.predictor, depth=min(replay.nstep, self.depth),
                                                             action_dim=self.action_dim, logs=logs)

            models = () if self.generate or not self.depth else (self.dynamics, self.projector, self.predictor)

            # Update critic, dynamics
            Utils.optimize(critic_loss + dynamics_loss,
                           self.critic, *models,
                           epoch=self.epoch if offline else self.episode)

        # Update encoder
        Utils.optimize(None,  # Using gradients from previous losses
                       self.encoder, epoch=self.epoch if offline else self.episode)

        if self.RL and not self.discrete:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, self.num_actions, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor, epoch=self.epoch if offline else self.episode)

        return logs
