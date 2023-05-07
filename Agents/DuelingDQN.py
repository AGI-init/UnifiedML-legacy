# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch.nn import Module

from Blocks.Architectures import MLP

from Utils import instantiate

from Agents import DQNAgent


class DuelingDQNAgent(DQNAgent):
    """
    Dueling Deep Q Networks Agent (https://arxiv.org/abs/1511.06581)
    """
    def __init__(self, hidden_dim, recipes, **kwargs):
        # For command line compatibility  e.g. python Run.py Agent=Agents.DuelingDQNAgent Pi_head=MLP +pi_head.depth=4
        Pi_head = recipes.actor.Pi_head

        class _DuelingDQN(DuelingDQN):
            """
            Dueling Architecture
            """
            def __init__(self, input_shape=50, output_shape=(2,)):
                super().__init__(input_shape, output_shape, Pi_head, hidden_dim)  # Assign Pi_head, hidden_dim

        # Use dueling architecture
        recipes.actor.Pi_head = _DuelingDQN

        super().__init__(hidden_dim=None, recipes=recipes, **kwargs)


class DuelingDQN(Module):
    """
    Dueling Architecture
    Can use with AC2Agent: python Run.py task=classify/mnist Pi_head=Agents.DuelingDQN.DuelingDQN
    """
    def __init__(self, input_shape=50, output_shape=(2,), Pi_head=None, hidden_dim=1024):
        super().__init__()

        # Default, MLP
        self.V = instantiate(Pi_head, input_shape=input_shape, output_shape=1) or MLP(input_shape, 1, hidden_dim, 2)

        self.A = instantiate(Pi_head, input_shape=input_shape,
                             output_shape=output_shape) or MLP(input_shape, output_shape, hidden_dim, 2)

    def forward(self, obs):
        # Value, Advantage
        V, A = self.V(obs), self.A(obs)

        # Q-Value
        return V + (A - A.mean())
