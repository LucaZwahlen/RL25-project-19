import argparse
from abc import ABC, abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from impoola_cnn.impoola.train.agents import PPOAgent
from impoola_cnn.impoola.utils.environment_knowledge import TEST_ENV_RANGE
from impoola_cnn.impoola.utils.noop_indices import get_noop_indices
from impoola_cnn.impoola.utils.success_rewards import get_success_reward

# Fix for numpy deprecations
if not hasattr(np, "bool"):
    np.bool = bool
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str


from impoola_cnn.impoola.maker.make_env import make_procgen_env


class SingleLevelVectorEnv:
    """
    Vectorized environment that internally manages 64 individual single-level environments.
    Drop-in replacement for your existing envs.step() interface.
    """

    def __init__(
        self,
        target_levels,
        env_id,
        distribution_mode="easy",
        seed=1,
        normalize_reward=False,
    ):
        """
        Args:
            target_levels: List of level IDs to create environments for (length should be 64)
            env_id: Environment name (e.g., "fruitbot")
            distribution_mode: "easy" or "hard"
            seed: Random seed
            normalize_reward: Whether to normalize rewards
        """
        self.target_levels = target_levels
        self.num_envs = len(target_levels)
        self.env_id = env_id
        self.distribution_mode = distribution_mode
        self.seed = seed
        self.normalize_reward = normalize_reward

        # Create individual environments
        self.envs = []
        for level_id in target_levels:
            fake_args = argparse.Namespace()
            fake_args.env_id = env_id
            fake_args.num_envs = 1
            fake_args.normalize_reward = normalize_reward
            fake_args.distribution_mode = distribution_mode

            single_env = make_procgen_env(
                fake_args,
                full_distribution=False,
                normalize_reward=normalize_reward,
                rand_seed=seed,
                render=False,
                distribution_mode=distribution_mode,
                num_levels_override=1,
                start_level_override=level_id,
            )
            self.envs.append(single_env)

        # Store environment info
        sample_env = self.envs[0]
        self.single_observation_space = sample_env.single_observation_space
        self.single_action_space = sample_env.single_action_space

        # Track episode completion info
        self._episode_info = {}
        self._current_level_seeds = np.array(
            target_levels
        )  # Track which level each env is on

    def reset(self):
        """Reset all environments and return stacked observations"""
        obs_list = []
        info_list = []

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        # Stack observations: (num_envs, channels, height, width)
        stacked_obs = np.concatenate(obs_list, axis=0)

        return stacked_obs, info_list

    def step(self, actions):
        """
        Step all environments with the given actions.

        Args:
            actions: numpy array of shape (num_envs,) with integer actions

        Returns:
            observations: (num_envs, channels, height, width)
            rewards: (num_envs,)
            terminated: (num_envs,)
            truncated: (num_envs,)
            info: dict with episode completion info
        """
        obs_list = []
        rewards_list = []
        terminated_list = []
        truncated_list = []

        # dict entries
        prev_level_seeds = np.zeros((len(self.envs)), dtype=np.int32)
        level_seeds = np.zeros((len(self.envs)), dtype=np.int32)
        prev_level_complete = np.zeros((len(self.envs)), dtype=bool)
        episode_r = np.zeros((len(self.envs)))
        episode_l = np.zeros((len(self.envs)))
        episode_t = np.zeros((len(self.envs)))

        episode_done = np.zeros((len(self.envs)), dtype=bool)

        # Step each environment individually
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(np.array([int(action)]))
            # Extract single-environment results
            obs_list.append(obs[0])
            rewards_list.append(reward[0])
            terminated_list.append(terminated[0])
            truncated_list.append(truncated[0])

            # combine the info dicts from the list of infos into one dict
            prev_level_seeds[i] = info["prev_level_seed"][0]
            level_seeds[i] = info["level_seed"][0]
            prev_level_complete[i] = info["prev_level_complete"][0]

            if "episode" in info:
                episode_r[i] = info["episode"]["r"][0]
                episode_l[i] = info["episode"]["l"][0]
                episode_t[i] = info["episode"]["t"][0]
            if "_episode" in info:
                episode_done[i] = info["_episode"][0]

        # Stack results
        # from list [3, 64, 64] to [64, 3, 64, 64]
        stacked_obs = np.array(obs_list)
        rewards_array = np.array(rewards_list)

        terminated_array = terminated_list
        truncated_array = truncated_list

        # Create info dict in the expected format
        info_dict = {}
        info_dict["prev_level_seed"] = prev_level_seeds
        info_dict["level_seed"] = level_seeds
        info_dict["prev_level_complete"] = prev_level_complete

        info_dict["episode"] = {"r": episode_r, "l": episode_l, "t": episode_t}
        if np.any(episode_done):
            info_dict["_episode"] = episode_done

        return stacked_obs, rewards_array, terminated_array, truncated_array, info_dict

    def close(self):
        """Close all individual environments"""
        for env in self.envs:
            env.close()

    @property
    def num_envs_property(self):
        return self.num_envs


def create_batch_iterator(total_levels, batch_size=64):
    """
    Generator that yields batches of level IDs for evaluation.

    Args:
        total_levels: Total number of levels to evaluate (e.g., TEST_ENV_RANGE)
        batch_size: Number of levels per batch

    Yields:
        List of level IDs for each batch
    """
    for start_idx in range(0, total_levels, batch_size):
        end_idx = min(start_idx + batch_size, total_levels)
        yield list(range(start_idx, end_idx))


class GenericActor(ABC):
    @abstractmethod
    def act(self, obs, eval_masks):
        pass


class ImpoolaPPOActor(GenericActor):
    def __init__(self, ppo_agent, device):
        self.ppo_agent = ppo_agent
        self.ppo_agent.to(device)
        self.ppo_agent.eval()
        self.device = device

    def act(self, obs):
        with torch.no_grad():
            action, _, _, _, _ = self.ppo_agent.get_action_and_value(
                obs
            )  # Remove deterministic parameter

        return action


SEED = 1
DISTRIBUTION_MODE = "easy"
NUM_ENVS = 256
ENV_ID = "chaser"


def eval(actor, device):

    checked_levels = np.zeros((TEST_ENV_RANGE,), dtype=bool)
    path_lengts = np.zeros(TEST_ENV_RANGE, dtype=np.int32)
    tick_lengts = np.zeros(TEST_ENV_RANGE, dtype=np.int32)
    successes = np.zeros(TEST_ENV_RANGE, dtype=bool)

    success_reward = torch.tensor(get_success_reward(ENV_ID), device=device)
    noop_indices = get_noop_indices(ENV_ID)
    noop_mask = torch.zeros((15,), dtype=torch.bool, device=device)
    for idx in noop_indices:
        noop_mask[idx] = 1

    batch_size = 128
    total_batches = (TEST_ENV_RANGE + batch_size - 1) // batch_size

    # progress bar
    pbar = tqdm(total=TEST_ENV_RANGE, desc="Evaluating on all-knowing path")
    env_level_ids = np.zeros(
        TEST_ENV_RANGE * 5, dtype=np.int32
    )  # just make it large so we dont run out of space lol
    envs_done = 0

    for batch_levels in create_batch_iterator(TEST_ENV_RANGE, batch_size):
        current_batch_size = len(batch_levels)
        envs = SingleLevelVectorEnv(
            target_levels=batch_levels,
            env_id=ENV_ID,
            distribution_mode=DISTRIBUTION_MODE,
            seed=SEED,
            normalize_reward=False,
        )

        obs, _ = envs.reset()
        obs = torch.tensor(obs, device=device)

        ticks = torch.zeros((current_batch_size,), device=device)
        steps = torch.zeros((current_batch_size,), device=device)
        success = torch.zeros((current_batch_size,), dtype=torch.bool, device=device)
        batch_done = torch.zeros((current_batch_size,), dtype=torch.bool, device=device)

        max_episode_steps = 1000  # fallback if some environemnts allow for infinte episodes that are bugged
        step_count = 0

        while not batch_done.all() and step_count < max_episode_steps:

            with torch.no_grad():
                action = actor.act(obs)  # Remove deterministic parameter

            obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            obs = torch.tensor(obs, device=device)

            rewards = torch.tensor(reward, device=device).view(-1)
            ticks += 1
            action_op_mask = ~(noop_mask[action])
            steps += action_op_mask
            success = success | (rewards >= success_reward - 0.25)

            batch_done = batch_done | torch.tensor(
                np.logical_or(terminated, truncated), device=device
            )

            if "_episode" in info.keys():
                done_mask = info["_episode"]
                completed_levels_prev_seeds = np.array(info["prev_level_seed"])[
                    done_mask
                ]

                path_lengts[completed_levels_prev_seeds] = (
                    steps[done_mask].cpu().numpy()
                )
                tick_lengts[completed_levels_prev_seeds] = (
                    ticks[done_mask].cpu().numpy()
                )
                successes[completed_levels_prev_seeds] = (
                    success[done_mask].cpu().numpy()
                )
                checked_levels[completed_levels_prev_seeds] = True

                ticks[done_mask] = 0
                steps[done_mask] = 0
                success[done_mask] = 0

                new_done = np.sum(done_mask).item()
                pbar.update(new_done)

                env_level_ids[envs_done : envs_done + new_done] = (
                    completed_levels_prev_seeds
                )

                envs_done += new_done

            step_count += 1

    envs.close()
    pbar.close()
    print(np.sum(checked_levels), "levels checked.")

    # save level_ids to readable file
    np.savetxt("all_knowing_level_ids.txt", env_level_ids, fmt="%d")
    np.savetxt("all_knowing_path_lengths.txt", path_lengts, fmt="%d")
    np.savetxt("all_knowing_tick_lengths.txt", tick_lengts, fmt="%d")
    np.savetxt("all_knowing_successes.txt", successes.astype(np.int32), fmt="%d")
    np.savetxt(
        "all_knowing_checked_levels.txt", checked_levels.astype(np.int32), fmt="%d"
    )


def load_impoola_ppo_checkpoint(
    checkpoint_path: str, device: torch.device, shape=(3, 64, 64), action_space=15
) -> PPOAgent:
    print(f"Loading checkpoint from {checkpoint_path}")
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # convert args from dict to namespace
    args_dict = checkpoint["args"]
    args = argparse.Namespace(**args_dict)

    envs = make_procgen_env(
        args,
        rand_seed=SEED,
        normalize_reward=args.normalize_reward,
        full_distribution=False,
    )
    print(args)
    agent = PPOAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale,
        out_features=args.latent_space_dim,
        cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,
    ).to(device)

    # Load weights exactly like train2.py does it
    agent.load_state_dict(checkpoint["agent_state_dict"])

    print(f"Loaded IMPOOLA model from step {checkpoint.get('step', 'unknown')}")

    return agent


# Standalone testing when run as main
if __name__ == "__main__":
    import os
    import sys

    # Add the train2.py arguments by importing them
    sys.path.append(os.path.dirname(__file__))
    from sit.train2 import parser  # This gets the parser with all train2.py arguments

    # Add test-specific arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )

    args = parser.parse_args()

    # Set device same as train2.py
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Load the trained agent
    agent = load_impoola_ppo_checkpoint(args.checkpoint, device)
    actor = ImpoolaPPOActor(agent, device)

    eval(actor, device)
