# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
import warnings

import torch
from torch.nn.functional import cross_entropy

from Blocks.Architectures import MLP
from Blocks.Architectures.Vision.ResNet import MiniResNet

import Utils

from Blocks.Augmentations import RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsemblePiActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class AC2Agent(torch.nn.Module):
    """Actor Critic Creator (AC2)
    RL, classification, generative modeling; online, offline; self-supervised learning; critic/actor ensembles;
    action space conversions; optimization schedules; EMA"""
    def __init__(self,
                 obs_spec, action_spec, num_actions, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 rand_steps, stddev_schedule,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 num_critics=2, num_actors=1, depth=0
                 ):
        super().__init__()

        self.discrete = discrete and not generate  # Discrete & Continuous supported!
        self.supervise = supervise  # And classification...
        self.RL = RL or generate
        self.generate = generate  # And generative modeling, too
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1
        self.ema = ema  # Can use an exponential moving average evaluation model

        self.num_actors = max(num_critics, num_actors) if self.discrete and self.RL else num_actors

        self.depth = depth  # Dynamics prediction depth

        # Image augmentation
        self.aug = Utils.instantiate(recipes.aug) or RandomShiftsAug(pad=4)

        # RL -> generate conversion
        if self.generate:
            standardize = False
            norm = True  # Normalize Obs to range [-1, 1]
            rand_steps = 0

            # Action = Imagined Obs
            action_spec.update({'shape': obs_spec.shape, 'discrete_bins': None,
                                'low': -1, 'high': 1, 'discrete': False})

            # Remove encoder, replace trunk with random noise
            recipes.encoder.Eyes = torch.nn.Identity()  # Generate "imagines" — no need for " seeing " via Eyes
            recipes.actor.trunk = Utils.Rand(size=trunk_dim)  # Generator observes random Gaussian noise as input

        self.discrete_as_continuous = action_spec.discrete and not self.discrete

        # Discrete -> continuous conversion
        if self.discrete_as_continuous:
            # Normalizing actions to range [-1, 1] significantly helps continuous RL
            action_spec.low, action_spec.high = (-1, 1) if self.RL else (None, None)

        # Continuous -> discrete conversion
        if self.discrete and not action_spec.discrete:
            assert num_actions > 1, 'Num actions cannot be 1 when discrete; try the "num_actions=" flag (>1) to ' \
                                    'divide each action dimension into discrete bins, or specify "discrete=false".'

            action_spec.discrete_bins = num_actions  # Continuous env has no discrete bins by default, must specify

        self.encoder = CNNEncoder(obs_spec, standardize=standardize, norm=norm, **recipes.encoder, parallel=parallel,
                                  lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay, ema_decay=ema_decay)

        self.actor = EnsemblePiActor(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.actor,
                                     ensemble_size=self.num_actors, discrete=self.discrete,
                                     stddev_schedule=stddev_schedule, creator=recipes.creator, rand_steps=rand_steps,
                                     lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                     ema_decay=ema_decay * ema)

        # Dynamics
        if self.depth and not self.generate:
            shape = list(self.encoder.feature_shape)

            # Action -> One-Hot, if single-dim discrete, otherwise action shape
            self.action_dim = action_spec.discrete_bins if self.discrete and action_spec.shape == (1,) \
                else self.actor.num_actions * self.actor.action_dim if self.discrete_as_continuous \
                else self.actor.action_dim

            shape[0] += self.action_dim  # Predicting from obs and action

            resnet = MiniResNet(input_shape=shape, stride=1, dims=(32, self.encoder.feature_shape[0]), depths=(1,))

            self.dynamics = CNNEncoder(self.encoder.feature_shape, context_dim=self.action_dim,
                                       Eyes=resnet, parallel=parallel,
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
                                      ensemble_size=self.num_actors if self.discrete else num_critics,
                                      discrete=self.discrete, ignore_obs=self.generate,
                                      lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                      ema_decay=ema_decay)

        # "Birth"

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor, self.critic):
            obs = torch.as_tensor(obs, device=self.device).float()

            # Exponential moving average (EMA) shadows
            encoder = self.encoder.ema if self.ema else self.encoder
            actor = self.actor.ema if self.ema and not (self.RL and self.discrete) else self.actor

            # "See"
            obs = encoder(obs)

            # Act
            Pi = actor(obs, self.step)

            action = Pi.sample() if self.training \
                else Pi.best

            store = {}

            if self.training:
                self.step += 1
                self.frame += len(obs)

                # Creator may store distribution as action rather than sampled action
                if Pi.store is not None:
                    store = {'action': Pi.store.cpu().numpy()}

            return action, store

    # "Dream"
    def learn(self, replay):
        # "Recall"

        batch = replay.sample(trajectories=True)
        obs, action, reward, discount, next_obs, label, *traj, step, ids, meta = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r, traj_l = traj

        # "Perceive"

        # Augment, encode present
        obs = self.aug(obs)
        features = self.encoder(obs, pool=False)
        obs = self.encoder.pool(features)

        if replay.nstep > 0 and not self.generate:
            with torch.no_grad():
                # Augment, encode future
                next_obs = self.aug(next_obs)
                next_obs = self.encoder(next_obs)

        # "Journal Teachings"

        logs = {'time': time.time() - self.birthday, 'step': self.step, 'frame': self.frame,
                'episode': self.episode} if self.log else None

        # Online -> Offline conversion
        if replay.offline:
            self.step += 1
            self.frame += len(obs)
            self.epoch = logs['epoch'] = replay.epoch
            logs['step'] = self.step
            logs['frame'] += 1  # Offline is 1 behind Online in training loop
            logs.pop('episode')

        # "Acquire Wisdom"

        instruct = not self.generate and label.nelement()  # Are labels present?

        # Classification
        if (self.supervise or replay.offline) and instruct:
            # "Via Example" / "Parental Support" / "School"

            # Inference
            Pi = self.actor(obs)
            y_predicted = (Pi.All_Qs if self.discrete else Pi.mean).mean(1)  # Average over ensembles

            # Cross entropy error
            error = cross_entropy(y_predicted, label.long(),
                                  reduction='none' if self.RL and replay.offline else 'mean')

            # Accuracy computation
            if self.log or self.RL and replay.offline:
                index = y_predicted.argmax(1, keepdim=True)  # Predicted class
                correct = (index.squeeze(1) == label).float()
                accuracy = correct.mean()

                if self.log:
                    logs.update({'accuracy': accuracy})

            # Supervised learning
            if self.supervise:
                # Supervised loss
                supervised_loss = error.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.actor, epoch=self.epoch if replay.offline else self.episode, retain_graph=True)

                if self.log:
                    logs.update({'supervised_loss': supervised_loss})

        # Reinforcement learning / generative modeling
        if self.RL:

            # Action and reward for supervised reinforcement learning
            if instruct:
                # "Via Feedback" / "Test Score" / "Letter Grade"

                if replay.offline:
                    action = (index if self.discrete else y_predicted).detach()
                    reward = correct if self.discrete else -error.detach()  # reward = -error
                else:  # Use Replay action from Online training
                    reward = (action.squeeze(1) == label).float() if self.discrete \
                        else -cross_entropy(action.squeeze(1), label.long(), reduction='none')  # reward = -error

            # Generative modeling
            if self.generate:
                # "Imagine"

                action, reward = obs, torch.ones(len(obs), 1, device=self.device)  # Discriminate Real

                # Critic loss
                critic_loss = QLearning.ensembleQLearning(self.critic, self.actor, obs, action, reward)

                # Update discriminator
                Utils.optimize(critic_loss, self.critic, epoch=self.epoch if replay.offline else self.episode)

                next_obs = None

                Pi = self.actor(obs)
                generated_image = Pi.best.flatten(1)  # Imagined image

                action, reward = generated_image, torch.zeros_like(reward)  # Discriminate Fake

            # Update reward log
            if self.log:
                logs.update({'reward': reward})

            # "Discern" / "Discriminate"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor, obs, action, reward, discount, next_obs,
                                                      self.step, logs=logs)

            # "Foretell"

            # Can only predict dynamics from available trajectories
            if self.depth > replay.nstep:
                warnings.warn(f"Dynamics 'depth' cannot exceed trajectory 'nstep'. Lowering 'depth' to {replay.nstep}. "
                              f"You can increase 'nstep' with the 'nstep={self.depth}' flag.")
                self.depth = replay.nstep

            # Dynamics loss
            dynamics_loss = 0 if self.depth == 0 or self.generate \
                else SelfSupervisedLearning.dynamicsLearning(features, traj_o, traj_a, traj_r,
                                                             self.encoder, self.dynamics, self.projector, self.predictor,
                                                             depth=self.depth, action_dim=self.action_dim, logs=logs)

            models = () if self.generate or not self.depth else (self.dynamics, self.projector, self.predictor)

            # "Sharpen Foresight"

            # Update critic, dynamics
            Utils.optimize(critic_loss + dynamics_loss, self.critic, *models, retain_graph=self.generate,
                           epoch=self.epoch if replay.offline else self.episode)

        # Update encoder
        Utils.optimize(None,  # Using gradients from previous losses
                       self.encoder, epoch=self.epoch if replay.offline else self.episode)

        if self.RL and not self.discrete:
            # "Change, Grow,  Ascend"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(), action,
                                                           self.step, logs=logs)

            # Update actor
            Utils.optimize(actor_loss, self.actor, epoch=self.epoch if replay.offline else self.episode)

        return logs
