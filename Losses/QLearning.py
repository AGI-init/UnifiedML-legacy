# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

import Utils


def ensembleQLearning(critic, actor, obs, action, reward, discount, next_obs, step, num_actions=1, logs=None):
    # Non-NaN next_obs
    has_future = ~torch.isnan(next_obs.flatten(1)[:, :1]).squeeze(1) * bool(next_obs.size(1))
    next_obs = next_obs[has_future]

    # Compute Bellman target
    with torch.no_grad():
        # Current reward
        target_Q = reward

        # Future action and Q-values
        next_action = All_Next_Qs = None

        # Discounted future reward
        if has_future.any():
            # Get actions for next_obs
            next_Pi = actor(next_obs, step)

            # Discrete Critic tabulates all actions for discrete envs a priori, no need to sample subset
            all_actions_known = hasattr(critic, 'action')

            if not all_actions_known:
                next_action = next_Pi.sample(num_actions)  # Sample actions

            if actor.discrete:
                All_Next_Qs = next_Pi.All_Qs  # Discrete Actor policy already knows all Q-values

            # Q-values per action
            next_Qs = critic.ema(next_obs, next_action, All_Next_Qs)  # Call a delayed-copy (EMA) of Critic: Q(obs, a)
            next_q = next_Qs.min(1)[0]  # Min-reduced ensemble
            next_q_norm = next_q - next_q.max(-1, keepdim=True)[0]  # Normalized

            # Weigh each action's Q-value by its probability
            temp = Utils.schedule(actor.stddev_schedule, step)  # Softmax temperature / "entropy"
            next_action_probs = (next_q_norm / temp).softmax(-1)  # Action probabilities
            next_v = torch.zeros_like(discount)
            next_v[has_future] = (next_q * next_action_probs).sum(-1, keepdim=True)  # Expected Q-value = E_a[Q(obs, a)]

            target_Q += discount * next_v

    Qs = critic(obs, action)  # Q-ensemble

    # Temporal difference (TD) error (via MSE, but could also use Huber)
    q_loss = F.mse_loss(Qs, target_Q.unsqueeze(1).expand_as(Qs))

    if logs is not None:
        logs['temporal_difference_error'] = q_loss
        logs.update({f'q{i}': Qs[:, i].median() for i in range(Qs.shape[1])})
        logs['target_q'] = target_Q.mean()

    return q_loss
