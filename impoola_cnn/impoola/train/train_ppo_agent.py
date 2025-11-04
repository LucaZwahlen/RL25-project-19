# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import csv
import os
import time
from collections import deque
from copy import deepcopy

from impoola_cnn.impoola.utils.evaluate_test_performance import evaluate_test_performance
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from impoola_cnn.impoola.utils.csv_logging import _evaluate


def train_ppo_agent(args, envs, agent, optimizer, device):
    """ Train the PPO agent """

    # Track training episode statistics
    training_episode_rewards = deque(maxlen=100)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.bool)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    logits = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), device=device)

    gamma = torch.tensor(args.gamma, device=device)
    gae_lambda = torch.tensor(args.gae_lambda, device=device)
    clip_coef = torch.tensor(args.clip_coef, device=device)
    norm_adv = torch.tensor(args.norm_adv, device=device)
    ent_coef = torch.tensor(args.ent_coef, device=device)
    vf_coef = torch.tensor(args.vf_coef, device=device)
    clip_vloss = torch.tensor(args.clip_vloss, device=device)
    learning_rate = optimizer.param_groups[0]["lr"].clone()
    max_grad_norm = torch.tensor(args.max_grad_norm, device=device)

    from impoola_cnn.impoola.train.ppo_criterion import ppo_gae, ppo_loss

    # TRY NOT TO MODIFY: start the game
    global_step = 0

    # Initialize cumulative training timer (excluding evaluation time)
    cumulative_training_time = 0.0
    iteration_start_time = time.time()

    next_obs, _ = envs.reset()

    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)

    for iteration in trange(1, args.num_iterations + 1):

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, pi_logits = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                logits[step] = pi_logits
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminated, truncated)

            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(next_obs, device=device)
            next_done = torch.tensor(next_done, device=device, dtype=torch.bool)

            # Collect training episode rewards
            if "_episode" in info.keys():
                completed_episodes = info["episode"]["r"][info["_episode"]]
                training_episode_rewards.extend(completed_episodes)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_values = values.reshape(-1)

        advantages, returns = ppo_gae(agent, next_done, next_obs, rewards, dones, values, gamma, gae_lambda, device,
                                      args.num_steps)

        advantages = advantages.clone()
        returns = returns.clone()

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0

        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                loss, pg_loss, v_loss, entropy_loss, logratio, ratio = ppo_loss(
                    agent,
                    b_obs[mb_inds],
                    b_logprobs[mb_inds], b_actions[mb_inds],
                    b_values[mb_inds], b_returns[mb_inds],
                    b_advantages[mb_inds],
                    b_advantages[mb_inds],
                    norm_adv, clip_coef, ent_coef, vf_coef, clip_vloss
                )

                total_policy_loss += pg_loss.item()
                total_value_loss += v_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

            eval_interval = max(1, args.num_iterations // args.n_datapoints_csv) if args.n_datapoints_csv else 1
            do_eval = args.num_iterations == 0 or (iteration % eval_interval == 0) or (iteration == args.num_iterations)
            if do_eval:
                iteration_start_time, cumulative_training_time = _evaluate(
                    total_policy_loss,
                    total_value_loss,
                    total_entropy_loss,
                    n_updates,
                    iteration,
                    global_step,
                    training_episode_rewards,
                    agent,
                    args,
                    device,
                    args.output_dir,
                    iteration_start_time,
                    cumulative_training_time,
                )

    return envs, agent, global_step, b_obs
