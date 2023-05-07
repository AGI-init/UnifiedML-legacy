# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from math import inf

from Blocks.Creator import MonteCarlo

from Utils import one_hot

from Agents import DQNAgent


class HardDQNAgent(DQNAgent):
    """Hard-Deep Q-Learning as in the original Nature paper"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.actor.creator.policy = HardDQNPolicy


class HardDQNPolicy(MonteCarlo):
    """
    A policy where returned probabilities for Q-learning correspond to a hard exploitation policy
    In Q Learning, the expected future Q-value depends on the probability of future actions; can be deterministic
    Can use with AC2Agent: python Run.py Policy=Agents.HardDQN.HardDQNPolicy
    """
    def log_prob(self, action=None):
        if action is None:
            num_actions = self.All_Qs.size(-2)
            log_prob = one_hot(self.best.squeeze(-1), num_actions, null_value=-inf)  # One-hot prob [0, ..., 1, ..., 0]

            return log_prob  # Learn deterministic Q-value-target
        else:
            return super().log_prob(action)  # For compatibility with continuous spaces/Agents
