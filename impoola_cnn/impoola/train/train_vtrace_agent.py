import csv
import os
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from impoola_cnn.impoola.eval.evaluation import (_get_game_range,
                                                 run_test_track,
                                                 run_training_track)
from impoola_cnn.impoola.prune.redo import run_redo
from impoola_cnn.impoola.train.train_ppo_agent import log_metrics_to_csv, evaluate_test_performance, log_sit_style_csv, \
    save_checkpoint_during_training
from impoola_cnn.impoola.train.vtrace_criterion import compute_vtrace_targets


def train_vtrace_agent(args, envs, agent, optimizer, device):
    postfix = ""
    game_range = _get_game_range(args.env_id)

    training_episode_rewards = deque(maxlen=100)

    output_dir = getattr(args, 'output_dir', 'outputs')
    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    sit_style_metrics_file = os.path.join(output_dir, "sit_style_metrics.csv")

    checkpoint_intervals = [int(args.num_iterations * i / 10) for i in range(1, 10)]

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
    pruning_steps_done = 0

    cumulative_training_time = 0.0
    iteration_start_time = time.time()

    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(N, device=device, dtype=torch.bool)

    redo_dict = run_redo(next_obs[:args.minibatch_size], agent, optimizer, args.redo_tau, False, False)

    initial_metrics = {
        "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
        "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
    }
    for i, (k, v) in enumerate(redo_dict['dormant_neurons_per_layer'].items()):
        initial_metrics[f"dormant_neurons/{i}_{k}"] = v
    log_metrics_to_csv(metrics_file, global_step, initial_metrics)

    if args.pruning_type != "Baseline":
        from impoola_cnn.impoola.maker.make_pruner import make_pruner
        from impoola_cnn.impoola.prune.pruning_func import pruning_step
        from impoola_cnn.impoola.utils.utils import calculate_global_parameters_number

        pruner, pruning_func, zero_weight_mode = make_pruner(args, agent, args.num_iterations)
        base_network_params = current_network_params = calculate_global_parameters_number(agent, zero_weight_mode=zero_weight_mode)
        global_sparsity = 0

    for iteration in trange(1, args.num_iterations + 1):
        start_time = time.time()

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

        avg_policy_loss = policy_loss.item()
        avg_value_loss = value_loss.item()
        avg_entropy_loss = entropy_loss.item()

        y_pred = target_values.flatten().detach().cpu().numpy()
        y_true = vs.flatten().detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        epoch_metrics = {
            "train/nupdates": iteration,
            "train/total_num_steps": global_step,
            "losses/action_loss": avg_policy_loss,
            "losses/value_loss": avg_value_loss,
            "losses/dist_entropy": avg_entropy_loss,
        }

        if len(training_episode_rewards) > 0:
            epoch_metrics.update({
                "train/mean_episode_reward": np.mean(training_episode_rewards),
                "train/median_episode_reward": np.median(training_episode_rewards),
            })

        iteration_end_time = time.time()
        cumulative_training_time += (iteration_end_time - iteration_start_time)

        test_mean, test_median = evaluate_test_performance(agent, args, device)

        epoch_metrics.update({
            "test/mean_episode_reward": test_mean,
            "test/median_episode_reward": test_median,
        })

        log_metrics_to_csv(metrics_file, global_step, epoch_metrics)

        if iteration % args.log_interval == 0 and len(training_episode_rewards) > 1:
            additional_metrics = {
                "train/fps": int(global_step / (time.time() - start_time)),
                "train/learning_rate": optimizer.param_groups[0]["lr"].item(),
                "train/value_loss": avg_value_loss,
                "train/policy_loss": avg_policy_loss,
                "train/entropy": avg_entropy_loss,
                "train/explained_variance": explained_var,
            }
            log_metrics_to_csv(metrics_file, global_step, additional_metrics)

        train_mean_reward = np.mean(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0
        train_median_reward = np.median(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0

        log_sit_style_csv(
            os.path.join(output_dir, "sit_format.csv"),
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

        if iteration % max(1, args.num_iterations // 20) == 0:
            if len(training_episode_rewards) > 0:
                print(f"\nUpdate {iteration}, step {global_step}")
                print(f"Last {len(training_episode_rewards)} training episodes: mean/median reward {np.mean(training_episode_rewards):.1f}/{np.median(training_episode_rewards):.1f}")
                print(f"Test episodes: mean/median reward {test_mean:.1f}/{test_median:.1f}")

        if iteration in checkpoint_intervals:
            progress = int((iteration / args.num_iterations) * 100)
            checkpoint_name = f"checkpoint_{progress:03d}_{iteration}"
            save_checkpoint_during_training(agent, optimizer, args, global_step, envs, output_dir, checkpoint_name)

        if iteration % max(1, args.num_iterations // 10) == 0:
            redo_dict = run_redo(b_obs[:args.minibatch_size], agent, optimizer, args.redo_tau, False, False)

            dormant_metrics = {
                "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
                "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
            }
            for i, (k, v) in enumerate(redo_dict['dormant_neurons_per_layer'].items()):
                dormant_metrics[f"dormant_neurons/{i}_{k}"] = v
            log_metrics_to_csv(metrics_file, global_step, dormant_metrics)

            eval_args = deepcopy(args)
            eval_args.n_episodes_rollout = int(1e3)
            run_training_track(agent, eval_args, global_step)
            run_test_track(agent, eval_args, global_step)

        if args.pruning_type == "UnstructuredNorm":
            did_prune, current_network_params, global_sparsity = pruning_step(
                args, agent, optimizer, pruning_func, pruner, iteration,
                zero_weight_mode, base_network_params, current_network_params, global_sparsity,
                b_obs,
            )

    return envs, agent, global_step, b_obs
