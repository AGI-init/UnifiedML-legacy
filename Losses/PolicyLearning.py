# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.


def deepPolicyGradient(actor, critic, obs, action=None, step=1, logs=None):

    if action is None or not action.requires_grad:  # If None or not differentiable
        action = actor(obs, step).mean  # Differentiable action ensemble

    Qs = critic(obs, action)

    q, _ = Qs.min(1)  # Min-reduced critic ensemble

    # When Sigmoid-activated
    if critic.binary:
        q = q.log()

    # Policy gradient ascent
    policy_loss = -q.mean()

    if logs is not None:
        logs['policy_loss'] = policy_loss

    return policy_loss
