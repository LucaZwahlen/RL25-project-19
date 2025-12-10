from copy import deepcopy

import numpy as np
import torch
from procgen import ProcgenEnv

from impoola_cnn.impoola.maker.make_env import make_procgen_env
from impoola_cnn.impoola.utils.csv_logging import EpisodeQueueCalculator
from impoola_cnn.impoola.utils.environment_knowledge import TEST_ENV_RANGE
from sit.baselines.common.vec_env.vec_monitor import VecMonitor
from sit.baselines.common.vec_env.vec_normalize import VecNormalize
from sit.baselines.common.vec_env.vec_remove_dict_obs import VecExtractDictObs
from sit.ucb_rl2_meta.envs import VecPyTorchProcgen


def evaluate(args, actor_critic, device, num_processes=64, aug_id=None):
    """Quick evaluation on test distribution - simplified version"""

    # Create test environment with full distribution
    test_args = deepcopy(args)
    test_args.num_envs = num_processes
    test_args.env_id = args.env_name
    test_args.distribution_mode = args.distribution_mode
    test_args.seed = args.seed

    test_envs = make_procgen_env(
        test_args,
        full_distribution=False,
        normalize_reward=False,
        rand_seed=test_args.seed,
        render=False,
        distribution_mode=test_args.distribution_mode,
        num_levels_override=TEST_ENV_RANGE - 1000,
        start_level_override=1000,
    )

    episodeQueueCalculator = EpisodeQueueCalculator(
        "test",
        test_args.seed,
        False,
        0,
        test_args.env_id,
        test_args.num_envs,
        test_args.distribution_mode,
        device,
    )

    num_episodes = 0
    target_episodes = 64

    obs, _ = test_envs.reset()
    obs = torch.tensor(obs, device=device, dtype=torch.float32)
    obs = obs / 255.0

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device
    )
    eval_masks = torch.ones(num_processes, 1, device=device)

    while num_episodes < target_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=False
            )

        obs, reward, terminated, truncated, info = test_envs.step(
            action.cpu().numpy().squeeze(-1)
        )
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        obs = obs / 255.0

        episodeQueueCalculator.update(
            action.squeeze(-1), torch.tensor(reward, device=device).view(-1)
        )
        done = np.logical_or(terminated, truncated)
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        if "_episode" in info.keys():
            episodeQueueCalculator.extend(info)
            done_mask = info["_episode"]
            num_episodes += np.sum(done_mask).item()

    test_envs.close()

    return (
        episodeQueueCalculator.get_statistics(),
        episodeQueueCalculator.get_raw_counts(),
    )


def evaluate_old(args, actor_critic, device, num_processes=32, aug_id=None):
    actor_critic.eval()

    # Sample Levels From the Full Distribution
    venv = ProcgenEnv(
        num_envs=num_processes,
        env_name=args.env_name,
        num_levels=0,
        start_level=0,
        distribution_mode=args.distribution_mode,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    eval_envs = VecPyTorchProcgen(venv, device)

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device
    )
    eval_masks = torch.ones(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 32:
        with torch.no_grad():
            if aug_id:
                obs = aug_id(obs)
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=False
            )

        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()

    print(
        "Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n".format(
            len(eval_episode_rewards),
            np.mean(eval_episode_rewards),
            np.median(eval_episode_rewards),
        )
    )

    return eval_episode_rewards
