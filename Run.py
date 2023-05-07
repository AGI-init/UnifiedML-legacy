# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import hydra
from hydra.utils import instantiate, call

import Utils


# Hydra conveniently and cleanly manages sys args
# Hyper-param arg files located in ./Hyperparams

@hydra.main(config_path='Hyperparams', config_name='args')
def main(args):

    # Set random seeds, device
    Utils.init(args)

    # Train, test environments
    env = instantiate(args.environment)
    generalize = instantiate(args.environment, train=False, seed=args.seed + 1234)

    for arg in ('obs_spec', 'action_spec', 'evaluate_episodes'):
        if hasattr(generalize.env, arg):
            setattr(args, arg, getattr(generalize.env, arg))

    # Agent
    agent = Utils.load(args.load_path, args.device, args.agent) if args.load \
        else instantiate(args.agent).to(args.device)

    train_steps = args.train_steps + agent.step

    # Experience replay
    replay = instantiate(args.replay,
                         meta_shape=getattr(agent, 'meta_shape', [0]))  # Optional agent-specific metadata can be stored

    # Loggers
    logger = instantiate(args.logger)

    vlogger = instantiate(args.vlogger) if args.log_media else None

    # Start
    converged = training = False
    while True:
        # Evaluate
        if converged or args.evaluate_per_steps and agent.step % args.evaluate_per_steps == 0 or args.train_steps == 0:

            for _ in range(args.generate or args.evaluate_episodes):
                exp, logs, vlogs = generalize.rollout(agent.eval(),  # agent.eval() just sets agent.training to False
                                                      vlog=args.log_media)

                logger.log(logs, 'Eval', exp if converged or args.train_steps == 0 else None)

            logger.dump_logs('Eval')

            if args.log_media:
                vlogger.dump(vlogs, f'{agent.step}')

        if args.plot_per_steps and (agent.step + 1) % args.plot_per_steps == 0 and not args.generate:
            call(args.plotting)

        if converged or args.train_steps == 0:
            break

        # Rollout
        experiences, logs, _ = env.rollout(agent.train(), steps=1)  # agent.train() just sets agent.training to True

        replay.add(experiences)

        if env.episode_done:
            if args.log_per_episodes and (agent.episode - 2 * replay.offline) % args.log_per_episodes == 0:
                logger.log(logs, 'Train' if training else 'Seed', dump=True)

            replay.add(store=not args.stream and env.last_episode_len > args.nstep)  # Only store full episodes
            replay.clear()

        converged = agent.step >= train_steps
        training = training or agent.step > args.seed_steps and len(replay) >= args.num_workers or replay.offline

        # Train agent
        if training and (args.learn_per_steps and agent.step % args.learn_per_steps == 0 or converged):

            for _ in range(args.learn_steps_after if converged else 1):  # Additional updates after all rollouts
                logs = agent.learn(replay)  # Learn
                if args.mixed_precision:
                    Utils.MP.update()  # For training speedup via automatic mixed precision

                if args.log_per_episodes:
                    logger.log(logs, 'Train')

        if training and args.save_per_steps and agent.step % args.save_per_steps == 0 or (converged and args.save):
            Utils.save(args.save_path, agent, args.agent, 'frame', 'step', 'episode', 'epoch')

        if training and args.load_per_steps and agent.step % args.load_per_steps == 0:
            agent = Utils.load(args.load_path, args.device, args.agent, ['frame', 'step', 'episode', 'epoch'], True)


if __name__ == '__main__':
    main()
