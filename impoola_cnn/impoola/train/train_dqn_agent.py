import csv
import os
import random
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import trange

from impoola_cnn.impoola.eval.evaluation import (_get_game_range,
                                                 _get_normalized_score,
                                                 run_test_track,
                                                 run_training_track)
from impoola_cnn.impoola.prune.redo import run_redo
from impoola_cnn.impoola.utils.schedules import linear_schedule
from impoola_cnn.impoola.utils.utils import StopTimer


def log_metrics_to_csv(csv_file, global_step, metrics_dict):
    """Log metrics to CSV file"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for key, value in metrics_dict.items():
            writer.writerow([global_step, key, value])


def log_sit_style_csv(csv_file, td_loss, q_values, value_loss, test_mean, test_median, train_mean, train_median, nupdates, total_steps, training_time):
    """Log metrics to CSV file in exact SIT format: td_loss,q_values,value_loss,test_mean,test_median,train_mean,train_median,nupdates,total_steps,training_time"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([td_loss, q_values, value_loss, test_mean, test_median, train_mean, train_median, nupdates, total_steps, training_time])


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


def make_replay_buffer(args, envs, device):
    if args.prioritized_replay:
        from impoola_cnn.impoola.utils.replay_buffer import \
            SimplifiedPrioritizedMultiStepReplayBuffer as Buffer

        # from impoola_cnn.impoola.utils.replay_buffer import PrioritizedMultiStepReplayBuffer as Buffer
        replay_buffer = Buffer(
            args.buffer_size,
            envs.single_observation_space_gymnasium,
            envs.single_action_space_gymnasium,
            device,
            optimize_memory_usage=False,
            handle_timeout_termination=False,
            n_envs=envs.num_envs,
            n_steps=args.multi_step,
            gamma=args.gamma,
            alpha=0.5,
            beta=0.5,
            beta_increment_per_sampling=0  # (1 - 0.4) / (args.total_timesteps // args.num_envs),
        )
    else:
        if args.multi_step > 1:
            from impoola_cnn.impoola.utils.replay_buffer import \
                MultiStepReplayBuffer as Buffer
            replay_buffer = Buffer(
                args.buffer_size,
                envs.single_observation_space_gymnasium,
                envs.single_action_space_gymnasium,
                device,
                optimize_memory_usage=False,
                handle_timeout_termination=False,
                n_envs=envs.num_envs,
                n_steps=args.multi_step,
                gamma=args.gamma,
            )
        else:
            replay_buffer = ReplayBuffer(
                args.buffer_size,
                envs.single_observation_space_gymnasium,
                envs.single_action_space_gymnasium,
                device,
                optimize_memory_usage=False,
                handle_timeout_termination=False,
                n_envs=envs.num_envs,
            )
    return replay_buffer


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
        target_episodes = 64  # Increased for more robust evaluation (~1 second)

        obs, _ = test_envs.reset()
        obs = torch.tensor(obs, device=device)

        while num_episodes < target_episodes:
            with torch.no_grad():
                q_values = agent(obs)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            obs, reward, terminated, truncated, info = test_envs.step(actions)
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


def train_dqn_agent(args, envs, agent, optimizer, device):
    """ Train the DQN agent """
    q_network, target_network = agent
    game_range = _get_game_range(args.env_id)

    # Track training episode statistics
    training_episode_rewards = deque(maxlen=100)

    # Setup CSV logging - assume output_dir and metrics_file are passed through args
    output_dir = getattr(args, 'output_dir', 'outputs')
    metrics_file = os.path.join(output_dir, "training_metrics.csv")

    # Calculate checkpoint intervals (every 10% of training)
    checkpoint_intervals = [int(args.total_timesteps * i / 10) for i in range(1, 10)]

    # Track update count for SIT logging and cumulative training time
    update_count = 0
    cumulative_training_time = 0.0
    training_start_time = time.time()

    if args.prioritized_replay:
        def unwrap_data(data):
            data, replay_indices, replay_weights = data
            return data, replay_indices, replay_weights
    else:
        def unwrap_data(data):
            return data, None, torch.ones(data.rewards.shape, device=device)

    if args.double_dqn:
        def get_target_max(target_network, next_observations, actions):
            online_actions = q_network(next_observations).argmax(dim=1, keepdim=True)
            target_max = target_network(next_observations)
            target_max = torch.gather(target_max, 1, online_actions).squeeze()
            return target_max
    else:
        def get_target_max(target_network, next_observations, actions):
            target_max = target_network(next_observations).max(dim=1)
            return target_max

    def update_dqn(data, args, q_network, target_network, optimizer):
        with torch.no_grad():
            data, replay_indices, replay_weights = unwrap_data(data)

            # if args.double_dqn:
            online_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
            target_max = target_network(data.next_observations)
            target_max = torch.gather(target_max, 1, online_actions).squeeze()
            # else:
            #     target_max, _ = target_network(data.next_observations).max(dim=1)
            # target_max = get_target_max(target_network, data.next_observations, data.actions)
            td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

        old_val = q_network(data.observations)
        old_val = torch.gather(old_val, 1, data.actions).squeeze()
        assert td_target.shape == old_val.shape, f"Shapes of td_target and old_val do not match: {td_target.shape} vs {old_val.shape}"

        td_error = td_target - old_val
        td_error = td_error * replay_weights
        loss = torch.mean(td_error ** 2)

        # optimize the model
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
        optimizer.step()

        return loss, old_val, td_error, replay_weights, replay_indices

    replay_buffer = make_replay_buffer(args, envs, device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    last_eval_step = -1  # Set to -1 to make sure to not evaluate at the very last step
    bar = trange(args.total_timesteps, initial=0, position=0)

    avg_returns = deque(maxlen=20)
    avg_sps = deque(maxlen=100)

    obs, _ = envs.reset()

    # Initial dormant neurons
    redo_dict = run_redo(torch.tensor(obs[:32], device=device), q_network, optimizer, args.redo_tau, False, False)

    initial_metrics = {
        "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
        "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
    }

    for i, (k, v) in enumerate(redo_dict['dormant_neurons_per_layer'].items()):
        initial_metrics[f"dormant_neurons/{i}_{k}"] = v

    log_metrics_to_csv(metrics_file, global_step, initial_metrics)

    duration_linear_schedule = args.exploration_fraction * args.total_timesteps

    stop_timer = StopTimer()
    stop_timer.start()

    while global_step < args.total_timesteps:
        start_time = time.perf_counter()

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, duration_linear_schedule, global_step)

        if args.anneal_lr:
            frac = 1.0 - (global_step - 1.0) / args.total_timesteps
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        # Exploration
        if args.softmax_exploration:
            with torch.inference_mode():
                q_values = q_network(torch.tensor(obs, device=device))
                temperature = 1.0
                action_probs = F.softmax(q_values / temperature, dim=1)
                actions = torch.multinomial(action_probs, num_samples=1).squeeze().cpu().numpy()
        else:
            if random.random() < epsilon:
                # TODO: we could sample for each environment separately
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():  # with torch.inference_mode():  <- issues with torch.compile
                    q_values = q_network(torch.tensor(obs, device=device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, info = envs.step(actions)
        next_done = np.logical_or(terminated, truncated)

        # update the bar by the taken step taking the number of environments into account
        global_step += envs.num_envs
        bar.n = global_step
        bar.refresh()

        # Collect training episode rewards for SIT logging
        if "_episode" in info.keys():
            completed_episodes = info["episode"]["r"][info["_episode"]]
            training_episode_rewards.extend(completed_episodes)

            episode_metrics = {
                f"charts/episodic_return": np.mean(info["episode"]["r"][info["_episode"]]),
                f"charts/episodic_return_normalized": _get_normalized_score(
                    info["episode"]["r"][info["_episode"]], game_range),
                f"charts/episodic_length": np.mean(info["episode"]["l"][info["_episode"]]),
            }
            log_metrics_to_csv(metrics_file, global_step, episode_metrics)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            if global_step % args.train_frequency == 0:

                # if train_frequency is smaller than one, we want to update more than once per step
                num_updates = int(1 / args.train_frequency) if args.train_frequency < 1 else int(args.train_frequency)

                for _ in range(num_updates):
                    data = replay_buffer.sample(args.batch_size)
                    loss, old_val, td_error, replay_weights, replay_indices = \
                        update_dqn(data, args, q_network, target_network, optimizer)

                    if args.prioritized_replay:
                        replay_buffer.update_priorities(replay_indices, td_error.detach().abs().cpu().numpy())
                        assert replay_weights.shape == td_error.shape, f"Replay_weights and td_error do not match"

                    update_count += 1

                training_metrics = {
                    f"losses/td_loss": loss,
                    f"losses/q_values": old_val.mean().item(),
                    f"charts/learning_rate": optimizer.param_groups[0]["lr"],
                    f"charts/epsilon": epsilon,
                    f"charts/replay_buffer_beta": replay_buffer.beta if args.prioritized_replay else 0.0,
                }

                log_metrics_to_csv(metrics_file, global_step, training_metrics)

                # SIT-style logging every N updates (like every epoch in PPO)
                if update_count % 16384 == 0:  # Log every 16k updates
                    # Get training time BEFORE test evaluation (excluding evaluation time)
                    current_training_time = stop_timer.get_elapsed_time()

                    # Quick test evaluation (like SIT) - EXCLUDE from training time
                    stop_timer.stop()  # Pause training timer during evaluation
                    test_mean, test_median = evaluate_test_performance(q_network, args, device)
                    stop_timer.start()  # Resume training timer after evaluation

                    # Calculate training performance
                    train_mean_reward = np.mean(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0
                    train_median_reward = np.median(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0

                    # Log in SIT format using training time BEFORE evaluation
                    log_sit_style_csv(
                        os.path.join(output_dir, "sit_format.csv"),
                        loss.item(),  # td_loss (equivalent to action_loss)
                        old_val.mean().item(),  # q_values (equivalent to dist_entropy)
                        0.0,  # value_loss (not applicable for DQN, set to 0)
                        test_mean,    # test_mean
                        test_median,  # test_median
                        train_mean_reward,  # train_mean
                        train_median_reward,  # train_median
                        update_count // 1000,  # nupdates (scaled down)
                        global_step,   # total_steps
                        current_training_time  # training_time (BEFORE test eval)
                    )

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_network_state_dict = q_network.state_dict()
                if args.tau is None:
                    target_network.load_state_dict(q_network_state_dict)
                else:
                    target_state_dict = target_network.state_dict()
                    for key, value in target_state_dict.items():
                        target_state_dict[key] = \
                            args.tau * q_network_state_dict[key] + (1.0 - args.tau) * target_state_dict[key]
                    target_network.load_state_dict(target_state_dict)

        replay_buffer.add(obs, next_obs, actions, rewards, terminated, info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        avg_sps.append(envs.num_envs / (time.perf_counter() - start_time))

        avg_sps_logs = {
            f"charts/avg_sps": int(np.mean(avg_sps))
        }
        log_metrics_to_csv(metrics_file, global_step, avg_sps_logs)

        if global_step >= last_eval_step + args.training_eval_ratio * args.total_timesteps:
            stop_timer.stop()

            # Estimate number of dormant neurons
            redo_dict = run_redo(torch.tensor(obs[:32], device=device), q_network, optimizer, args.redo_tau, False,
                                 False)

            dormant_metrics = {
                "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
                "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
            }

            for i, (k, v) in enumerate(redo_dict['dormant_neurons_per_layer'].items()):
                dormant_metrics[f"dormant_neurons/{i}_{k}"] = v

            log_metrics_to_csv(metrics_file, global_step, dormant_metrics)

            # calc_translation_sensitivity(q_network, torch.tensor(obs[:32], device=device), device)

            # eval
            eval_args = deepcopy(args)
            eval_args.n_episodes_rollout = int(1e3)
            run_training_track(q_network, eval_args, global_step)
            run_test_track(q_network, eval_args, global_step)

            last_eval_step = global_step
            stop_timer.start()

    avg_sps_time = {
        f"charts/avg_sps": int(np.mean(avg_sps)),
        f"charts/elapsed_train_time": stop_timer.get_elapsed_time(),
    }
    log_metrics_to_csv(metrics_file, global_step, avg_sps_time)
    # Free memory of replay buffer
    del replay_buffer
    return envs, q_network, global_step, torch.tensor(obs, device=device)
