from copy import deepcopy

import numpy as np
import torch

from impoola_cnn.impoola.utils.noop_indices import get_noop_indices


def evaluate_test_performance(agent, args, device):
    """Quick evaluation on test distribution - simplified version"""
    from impoola_cnn.impoola.maker.make_env import make_procgen_env

    try:
        # Create test environment with full distribution
        test_args = deepcopy(args)
        test_args.num_envs = 64

        test_envs = make_procgen_env(test_args, full_distribution=False, normalize_reward=False, rand_seed=args.seed, render=False,
                                     distribution_mode=args.distribution_mode, num_levels_override=10000, start_level_override=1000)

        noop_indices = get_noop_indices(test_args.env_id)
        noop_mask = torch.zeros((test_envs.single_action_space.n,), dtype=torch.bool, device=device)
        for idx in noop_indices:
            noop_mask[idx] = 1

        episode_rewards = []
        episode_level_seeds = []
        episode_num_ticks = []
        episode_num_steps = []

        N = test_args.num_envs
        ticks = torch.zeros((N,), device=device)
        steps = torch.zeros((N,), device=device)

        num_episodes = 0
        target_episodes = 64  # Increased from 32 for more robust evaluation (~1 second)

        obs, _ = test_envs.reset()
        obs = torch.tensor(obs, device=device)

        while num_episodes < target_episodes:
            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs)  # Remove deterministic parameter

            obs, reward, terminated, truncated, info = test_envs.step(action.cpu().numpy())
            obs = torch.tensor(obs, device=device)

            ticks += 1
            action_op_mask = ~(noop_mask[action])
            steps += action_op_mask

            if "_episode" in info.keys():
                completed_episodes = info["episode"]["r"][info["_episode"]]
                completed_levels_prev_seeds = np.array(info['prev_level_seed'])[info["_episode"]]
                episode_rewards.extend(completed_episodes)
                episode_level_seeds.extend(completed_levels_prev_seeds)
                completed_ticks = ticks[info["_episode"]].cpu().numpy()
                episode_num_ticks.extend(completed_ticks)
                completed_steps = steps[info["_episode"]].cpu().numpy()
                episode_num_steps.extend(completed_steps)
                ticks[info["_episode"]] = 0
                steps[info["_episode"]] = 0
                num_episodes = len(episode_rewards)

        test_envs.close()

        return np.mean(episode_rewards), np.median(episode_rewards), np.mean(episode_num_ticks), np.mean(episode_num_steps), episode_level_seeds

    except Exception as e:

        # rethrow
        raise e

        # Silently handle errors and return zeros
        return 0.0, 0.0, 0.0, 0.0, []
