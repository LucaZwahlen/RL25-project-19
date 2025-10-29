# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
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


def log_metrics_to_csv(csv_file, global_step, metrics_dict):
    """Log metrics to CSV file"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for key, value in metrics_dict.items():
            writer.writerow([global_step, key, value])


def log_sit_style_csv(csv_file, action_loss, dist_entropy, value_loss, test_mean, test_median, train_mean, train_median, nupdates, total_steps, training_time):
    """Log metrics to CSV file in exact SIT format: action_loss,dist_entropy,value_loss,test_mean,test_median,train_mean,train_median,nupdates,total_steps,training_time"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([action_loss, dist_entropy, value_loss, test_mean, test_median, train_mean, train_median, nupdates, total_steps, training_time])


def save_checkpoint_during_training(agent, optimizer, args, global_step, envs, output_dir, checkpoint_name):
    """Save model checkpoint during training"""
    checkpoint_path = os.path.join(output_dir, f"{checkpoint_name}.pt")
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'global_step': global_step,
        'obs_rms': getattr(envs, 'obs_rms', None),
        'return_rms': getattr(envs, 'return_rms', None),
    }, checkpoint_path)
    print(f"Training checkpoint saved: {checkpoint_path}")
    return checkpoint_path


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


def train_ppo_agent(args, envs, agent, optimizer, device):
    """ Train the PPO agent """
    postfix = ""
    game_range = _get_game_range(args.env_id)

    # Track training episode statistics
    training_episode_rewards = deque(maxlen=100)

    # Setup CSV logging - assume output_dir and metrics_file are passed through args
    output_dir = getattr(args, 'output_dir', 'outputs')
    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    sit_style_metrics_file = os.path.join(output_dir, "sit_style_metrics.csv")

    # Calculate checkpoint intervals (every 10% of training)
    checkpoint_intervals = [int(args.num_iterations * i / 10) for i in range(1, 10)]

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
    pruning_steps_done = 0

    # Initialize cumulative training timer (excluding evaluation time)
    cumulative_training_time = 0.0
    iteration_start_time = time.time()

    next_obs, _ = envs.reset()

    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)

    # Initial dormant neurons
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
        from impoola_cnn.impoola.utils.utils import \
            calculate_global_parameters_number

        pruner, pruning_func, zero_weight_mode = make_pruner(args, agent, args.num_iterations)
        base_network_params = current_network_params = \
            calculate_global_parameters_number(agent, zero_weight_mode=zero_weight_mode)
        global_sparsity = 0

    for iteration in trange(1, args.num_iterations + 1):
        start_time = time.time()

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
        clipfracs = []
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Calculate average losses
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy_loss = total_entropy_loss / n_updates

        # Calculate explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # SIT-style logging every epoch
        epoch_metrics = {
            "train/nupdates": iteration,
            "train/total_num_steps": global_step,
            "losses/action_loss": avg_policy_loss,
            "losses/value_loss": avg_value_loss,
            "losses/dist_entropy": avg_entropy_loss,
        }

        # Add training performance if we have episode data
        if len(training_episode_rewards) > 0:
            epoch_metrics.update({
                "train/mean_episode_reward": np.mean(training_episode_rewards),
                "train/median_episode_reward": np.median(training_episode_rewards),
            })

        # Update cumulative training time (excluding this iteration's quick test eval)
        iteration_end_time = time.time()
        cumulative_training_time += (iteration_end_time - iteration_start_time)

        # Quick test evaluation every epoch (like SIT) - silently handle errors and EXCLUDE from training time
        test_mean, test_median = evaluate_test_performance(agent, args, device)
        # Note: We don't add eval time to cumulative_training_time

        epoch_metrics.update({
            "test/mean_episode_reward": test_mean,
            "test/median_episode_reward": test_median,
        })

        # Log to original CSV format (keep all original logging)
        log_metrics_to_csv(metrics_file, global_step, epoch_metrics)

        # Additional logging for original behavior (only at log_interval)
        if iteration % args.log_interval == 0 and len(training_episode_rewards) > 1:
            # Original detailed metrics logging
            additional_metrics = {
                "train/fps": int(global_step / (time.time() - start_time)),
                "train/learning_rate": optimizer.param_groups[0]["lr"].item(),
                "train/value_loss": avg_value_loss,
                "train/policy_loss": avg_policy_loss,
                "train/entropy": avg_entropy_loss,
                "train/old_approx_kl": old_approx_kl.item() if 'old_approx_kl' in locals() else 0,
                "train/approx_kl": approx_kl.item() if 'approx_kl' in locals() else 0,
                "train/clipfrac": np.mean(clipfracs) if clipfracs else 0,
                "train/explained_variance": explained_var,
            }
            log_metrics_to_csv(metrics_file, global_step, additional_metrics)

        # SIT-style CSV logging ONCE per epoch (exact format requested)
        train_mean_reward = np.mean(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0
        train_median_reward = np.median(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0

        log_sit_style_csv(
            os.path.join(output_dir, "sit_format.csv"),
            avg_policy_loss,  # action_loss
            avg_entropy_loss,  # dist_entropy
            avg_value_loss,   # value_loss
            test_mean,        # test_mean
            test_median,      # test_median
            train_mean_reward,  # train_mean
            train_median_reward,  # train_median
            iteration,        # nupdates
            global_step,      # total_steps
            cumulative_training_time  # training_time (cumulative, excluding evaluations)
        )

        # Start timing next iteration (after evaluation is complete)
        iteration_start_time = time.time()

        # Print progress (similar to SIT format) - only occasionally to reduce noise
        if iteration % max(1, args.num_iterations // 20) == 0:
            if len(training_episode_rewards) > 0:
                print(f"\nUpdate {iteration}, step {global_step}")
                print(f"Last {len(training_episode_rewards)} training episodes: mean/median reward {np.mean(training_episode_rewards):.1f}/{np.median(training_episode_rewards):.1f}")
                print(f"Test episodes: mean/median reward {test_mean:.1f}/{test_median:.1f}")

        # Save checkpoint every 10% of training
        if iteration in checkpoint_intervals:
            progress = int((iteration / args.num_iterations) * 100)
            checkpoint_name = f"checkpoint_{progress:03d}_{iteration}"
            save_checkpoint_during_training(agent, optimizer, args, global_step, envs, output_dir, checkpoint_name)

        # Detailed evaluation every 10% (keep original behavior for detailed tracking)
        if iteration % max(1, args.num_iterations // 10) == 0:

            # Estimate number of dormant neurons
            redo_dict = run_redo(b_obs[:args.minibatch_size], agent, optimizer, args.redo_tau, False, False)

            dormant_metrics = {
                "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
                "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
            }

            for i, (k, v) in enumerate(redo_dict['dormant_neurons_per_layer'].items()):
                dormant_metrics[f"dormant_neurons/{i}_{k}"] = v

            log_metrics_to_csv(metrics_file, global_step, dormant_metrics)

            # Detailed eval
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
