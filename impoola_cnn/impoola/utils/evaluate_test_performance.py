from copy import deepcopy

import numpy as np
import torch


def evaluate_test_performance(agent, args, device):
    """Quick evaluation on test distribution - simplified version"""
    from impoola_cnn.impoola.maker.make_env import make_an_env

    try:
        # Create test environment with full distribution
        test_args = deepcopy(args)
        test_args.num_envs = min(32, args.num_envs)  # Use more envs for better parallelization

        test_envs = make_an_env(test_args, seed=42, normalize_reward=False, full_distribution=True)

        episode_rewards = []
        num_episodes = 0
        target_episodes = 64  # Increased from 32 for more robust evaluation (~1 second)

        obs, _ = test_envs.reset()
        obs = torch.tensor(obs, device=device)

        while num_episodes < target_episodes:
            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs)  # Remove deterministic parameter

            obs, reward, terminated, truncated, info = test_envs.step(action.cpu().numpy())
            obs = torch.tensor(obs, device=device)

            if "_episode" in info.keys():
                completed_episodes = info["episode"]["r"][info["_episode"]]
                episode_rewards.extend(completed_episodes)
                num_episodes = len(episode_rewards)

        test_envs.close()

        if len(episode_rewards) == 0:
            return 0.0, 0.0

        return np.mean(episode_rewards), np.median(episode_rewards)

    except Exception as e:
        # Silently handle errors and return zeros
        return 0.0, 0.0
