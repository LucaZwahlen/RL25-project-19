import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from impoola_cnn.impoola.train.vtrace_criterion import compute_vtrace_targets
from impoola_cnn.impoola.utils.augmentation import Augmentation
from impoola_cnn.impoola.utils.csv_logging import log_sit_style_csv
from impoola_cnn.impoola.utils.evaluate_test_performance import evaluate_test_performance


def train_vtrace_agent(args, envs, agent, optimizer, device):
    training_episode_rewards = deque(maxlen=100)

    augment = Augmentation()

    T = args.unroll_length
    N = args.num_envs
    obs = torch.zeros((T, N) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((T, N) + envs.single_action_space.shape, device=device)
    rewards = torch.zeros((T, N), device=device)
    dones = torch.zeros((T, N), device=device, dtype=torch.bool)
    values = torch.zeros((T, N), device=device)
    behavior_logits = torch.zeros((T, N, envs.single_action_space.n), device=device)

    gamma = torch.tensor(args.gamma, device=device)
    ent_coef = torch.tensor(args.ent_coef, device=device)
    vf_coef = torch.tensor(args.vf_coef, device=device)
    rho_bar = torch.tensor(args.vtrace_rho_bar, device=device)
    c_bar = torch.tensor(args.vtrace_c_bar, device=device)
    learning_rate = optimizer.param_groups[0]["lr"].clone()
    max_grad_norm = torch.tensor(args.max_grad_norm, device=device)

    global_step = 0

    cumulative_training_time = 0.0
    iteration_start_time = time.time()

    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, device=device)
    if args.use_augmentation:
        next_obs = augment(next_obs)

    next_done = torch.zeros(N, device=device, dtype=torch.bool)

    for iteration in trange(1, args.num_iterations + 1):

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        for step in range(0, T):
            global_step += N
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, _, ent, value, pi_logits = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                behavior_logits[step] = pi_logits
            actions[step] = action

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminated, truncated)

            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(next_obs, device=device)
            next_done = torch.tensor(next_done, device=device, dtype=torch.bool)

            if "_episode" in info.keys():
                completed_episodes = info["episode"]["r"][info["_episode"]]
                training_episode_rewards.extend(completed_episodes)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        T_, N_ = T, N

        with torch.no_grad():
            bootstrap_value = agent.get_pi_and_value(next_obs)[1].squeeze(-1)

        flat_obs = b_obs
        target_pi, target_values_flat = agent.get_pi_and_value(flat_obs)
        target_logits_flat = target_pi.logits
        target_values = target_values_flat.reshape(T_, N_)
        target_logits = target_logits_flat.reshape(T_, N_, -1)

        vs, pg_adv = compute_vtrace_targets(
            rewards=rewards,
            dones=dones,
            values=values,
            bootstrap_value=bootstrap_value,
            behavior_logits=behavior_logits,
            target_logits=target_logits,
            gamma=gamma,
            rho_bar=rho_bar,
            c_bar=c_bar,
            actions=actions.reshape(T_ * N_)
        )

        flat_target_pi = torch.distributions.Categorical(logits=target_logits_flat)
        logp = flat_target_pi.log_prob(actions.reshape(T_ * N_)).reshape(T_, N_)
        entropy = flat_target_pi.entropy().reshape(T_, N_)

        policy_loss = -(pg_adv.detach() * logp).mean()
        value_loss = 0.5 * (target_values - vs.detach()).pow(2).mean()
        entropy_loss = entropy.mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        eval_interval = max(1, args.num_iterations // args.n_datapoints_csv) if args.n_datapoints_csv else 1
        do_eval = args.num_iterations == 0 or (iteration % eval_interval == 0) or (iteration == args.num_iterations)
        if do_eval:
            avg_policy_loss = policy_loss.item()
            avg_value_loss = value_loss.item()
            avg_entropy_loss = entropy_loss.item()

            iteration_end_time = time.time()
            cumulative_training_time += (iteration_end_time - iteration_start_time)

            test_mean, test_median = evaluate_test_performance(agent, args, device)

            train_mean_reward = np.mean(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0
            train_median_reward = np.median(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0

            log_sit_style_csv(
                os.path.join(args.output_dir, "sit_format.csv"),
                avg_policy_loss,
                avg_entropy_loss,
                avg_value_loss,
                test_mean,
                test_median,
                train_mean_reward,
                train_median_reward,
                iteration,
                global_step,
                cumulative_training_time
            )

            iteration_start_time = time.time()

    return envs, agent, global_step, b_obs
