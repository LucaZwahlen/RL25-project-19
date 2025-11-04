# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy

import numpy as np
import torch

from impoola_cnn.impoola.maker.make_env import progcen_hns


def rollout(envs, agent, n_episodes=10000, noise_scale=None, deterministic=True):
    device = next(agent.parameters()).device

    # We cannot simply append eps whenever one is ready because this would bias the evaluation towards eps that are fast
    eval_avg_return = []
    eps_to_do_per_env = np.zeros(envs.num_envs)
    for idx in range(n_episodes):
        eps_to_do_per_env[idx % envs.num_envs] += 1

    assert sum(eps_to_do_per_env) == n_episodes, f"Sum of eps_to_do_per_env is broken: {sum(eps_to_do_per_env)}"

    agent.eval()
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    obs_shape = next_obs.shape

    with torch.inference_mode():  # TODO: Can it be that this here influences the layer norm?
        while len(eval_avg_return) < n_episodes:
            action = agent.get_action(next_obs, deterministic=deterministic)
            next_obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())

            if noise_scale is not None:
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                noise = torch.randn(obs_shape, device=device) * noise_scale
                next_obs.add_(noise.round()).clamp_(0.0, 255.0)
            else:
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)

            if "_episode" in info.keys():
                for i in range(len(info["_episode"])):
                    if info["_episode"][i] and eps_to_do_per_env[i] > 0:
                        eval_avg_return.append(info["episode"]["r"][i])
                        eps_to_do_per_env[i] -= 1

    agent.train()
    return eval_avg_return


def _get_normalized_score(eval_avg_return, game_range):
    if game_range is not None:
        normalized_score = (np.mean(eval_avg_return) - game_range[1]) / (game_range[2] - game_range[1])
    else:
        normalized_score = np.mean(eval_avg_return)
    return normalized_score


def _get_game_range(env_id):
    for game_name, game_range in progcen_hns.items():
        if env_id in game_name:
            print(f"Game range: {game_range}")
            return game_range
    raise ValueError(f"Unknown environment: {env_id}")


def get_normalized_score(env_id, eval_avg_return):
    game_range = _get_game_range(env_id)
    return _get_normalized_score(eval_avg_return, game_range)
