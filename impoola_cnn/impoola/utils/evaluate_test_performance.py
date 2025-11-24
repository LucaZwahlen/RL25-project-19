import random
import time
from copy import deepcopy

import numpy as np
import torch

from impoola_cnn.impoola.maker.make_env import make_procgen_env
from impoola_cnn.impoola.utils.csv_logging import EpisodeQueueCalculator
from impoola_cnn.impoola.utils.environment_knowledge import TEST_ENV_RANGE


def get_action_and_value(args, agent, obs):
    """Generic action and value function for different agent types"""
    deterministic = args.deterministic_rollout if hasattr(args, 'deterministic_rollout') else False

    if hasattr(agent, 'get_action_and_value'):
        return agent.get_action_and_value(obs)
    else:
        action = agent.get_action(obs, deterministic=deterministic)
        return action, None, None, None, None


def evaluate_test_performance(agent, args, device, force_rdm_seed=False):
    """Quick evaluation on test distribution - simplified version"""

    # Create test environment with full distribution
    test_args = deepcopy(args)
    test_args.num_envs = 64
    rdm_seed = int(time.time()) % 1000000 if force_rdm_seed else args.seed
    test_envs = make_procgen_env(test_args, full_distribution=False, normalize_reward=False, rand_seed=rdm_seed, render=False,
                                 distribution_mode=args.distribution_mode, num_levels_override=TEST_ENV_RANGE - 1000, start_level_override=1000)

    episodeQueueCalculator = EpisodeQueueCalculator('test', args.seed, False, 0, test_args.env_id, test_args.num_envs, test_args.distribution_mode, device)

    num_episodes = 0
    target_episodes = 64

    obs, _ = test_envs.reset()
    obs = torch.tensor(obs, device=device)

    while num_episodes < target_episodes:
        with torch.no_grad():

            action, _, _, _, _ = get_action_and_value(test_args, agent, obs)

        obs, reward, terminated, truncated, info = test_envs.step(action.cpu().numpy())
        obs = torch.tensor(obs, device=device)

        episodeQueueCalculator.update(action, torch.tensor(reward, device=device).view(-1))

        if "_episode" in info.keys():
            episodeQueueCalculator.extend(info)
            done_mask = info["_episode"]
            num_episodes += np.sum(done_mask).item()

    test_envs.close()

    return episodeQueueCalculator.get_statistics(), episodeQueueCalculator.get_raw_counts()


def evaluate_test_performance_grpo(agent, args, device, force_rdm_seed=False):
    """Evaluation loop for pure GRPO agents (no critic, no value head)."""

    test_args = deepcopy(args)
    test_args.num_envs = 64
    rdm_seed = int(time.time()) % 1_000_000 if force_rdm_seed else args.seed

    # Test on held-out level range
    test_envs = make_procgen_env(
        test_args,
        full_distribution=False,
        normalize_reward=False,
        rand_seed=rdm_seed,
        render=False,
        distribution_mode=args.distribution_mode,
        num_levels_override=TEST_ENV_RANGE - 1000,
        start_level_override=1000,
    )

    episodeQueueCalculator = EpisodeQueueCalculator(
        'test',
        args.seed,
        False,
        0,
        test_args.env_id,
        test_args.num_envs,
        test_args.distribution_mode,
        device
    )

    num_episodes = 0
    target = 64

    obs, _ = test_envs.reset()
    obs = torch.tensor(obs, device=device)

    while num_episodes < target:
        with torch.no_grad():

            # policy only → logits → sample or argmax
            pi = agent.forward(obs)

            if getattr(test_args, "deterministic_rollout", False):
                action = torch.argmax(pi.logits, dim=-1)
            else:
                action = pi.sample()

        # Step through environment
        obs, reward, term, trunc, info = test_envs.step(action.cpu().numpy())
        obs = torch.tensor(obs, device=device)

        # Log rewards into calculator
        episodeQueueCalculator.update(action, torch.tensor(reward, device=device).view(-1))

        if "_episode" in info.keys():
            episodeQueueCalculator.extend(info)
            done_mask = info["_episode"]
            num_episodes += np.sum(done_mask).item()

    test_envs.close()

    return (
        episodeQueueCalculator.get_statistics(),
        episodeQueueCalculator.get_raw_counts(),
    )
