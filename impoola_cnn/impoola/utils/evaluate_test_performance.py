from copy import deepcopy

import numpy as np
import torch

from impoola_cnn.impoola.maker.make_env import make_procgen_env
from impoola_cnn.impoola.utils.csv_logging import EpisodeQueueCalculator
from impoola_cnn.impoola.utils.environment_knowledge import TEST_ENV_RANGE


def evaluate_test_performance(agent, args, device):
    """Quick evaluation on test distribution - simplified version"""

    # Create test environment with full distribution
    test_args = deepcopy(args)
    test_args.num_envs = 64

    test_envs = make_procgen_env(test_args, full_distribution=False, normalize_reward=False, rand_seed=args.seed, render=False,
                                 distribution_mode=args.distribution_mode, num_levels_override=TEST_ENV_RANGE - 1000, start_level_override=1000)

    episodeQueueCalculator = EpisodeQueueCalculator(False, False, 0, test_args.env_id, test_args.num_envs, test_args.distribution_mode, device)

    num_episodes = 0
    target_episodes = 64

    obs, _ = test_envs.reset()
    obs = torch.tensor(obs, device=device)

    while num_episodes < target_episodes:
        with torch.no_grad():
            action, _, _, _, _ = agent.get_action_and_value(obs)  # Remove deterministic parameter

        obs, reward, terminated, truncated, info = test_envs.step(action.cpu().numpy())
        obs = torch.tensor(obs, device=device)

        episodeQueueCalculator.update(action, torch.tensor(reward, device=device).view(-1))

        if "_episode" in info.keys():
            episodeQueueCalculator.extend(info)
            done_mask = info["_episode"]
            num_episodes += np.sum(done_mask).item()

    test_envs.close()

    return episodeQueueCalculator.get_statistics()
