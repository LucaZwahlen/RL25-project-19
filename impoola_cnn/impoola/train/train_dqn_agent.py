import os
import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from impoola_cnn.impoola.utils.csv_logging import log_sit_style_csv
from impoola_cnn.impoola.utils.evaluate_test_performance import evaluate_test_performance
from impoola_cnn.impoola.utils.schedules import linear_schedule
from impoola_cnn.impoola.utils.utils import StopTimer


def make_replay_buffer(args, envs, device):
    from impoola_cnn.impoola.utils.replay_buffer import \
        SimplifiedPrioritizedMultiStepReplayBuffer as Buffer

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
    return replay_buffer


def train_dqn_agent(args, envs, agent, optimizer, device):
    q_network, target_network = agent

    training_episode_rewards = deque(maxlen=100)

    update_count = 0

    def unwrap_data(data):
        data, replay_indices, replay_weights = data
        return data, replay_indices, replay_weights

    def update_dqn(data, args, q_network, target_network, optimizer):
        with torch.no_grad():
            data, replay_indices, replay_weights = unwrap_data(data)

            online_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
            target_max = target_network(data.next_observations)
            target_max = torch.gather(target_max, 1, online_actions).squeeze()
            td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

        old_val = q_network(data.observations)
        old_val = torch.gather(old_val, 1, data.actions).squeeze()
        assert td_target.shape == old_val.shape, f"Shapes of td_target and old_val do not match: {td_target.shape} vs {old_val.shape}"

        td_error = td_target - old_val
        td_error = td_error * replay_weights
        loss = torch.mean(td_error ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
        optimizer.step()

        return loss, old_val, td_error, replay_weights, replay_indices

    replay_buffer = make_replay_buffer(args, envs, device)

    global_step = 0
    bar = trange(args.total_timesteps, initial=0, position=0)

    obs, _ = envs.reset()

    duration_linear_schedule = args.exploration_fraction * args.total_timesteps

    stop_timer = StopTimer()
    stop_timer.start()

    while global_step < args.total_timesteps:
        epsilon = linear_schedule(args.start_e, args.end_e, duration_linear_schedule, global_step)

        if args.anneal_lr:
            frac = 1.0 - (global_step - 1.0) / args.total_timesteps
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        if args.softmax_exploration:
            with torch.inference_mode():
                q_values = q_network(torch.tensor(obs, device=device))
                temperature = 1.0
                action_probs = F.softmax(q_values / temperature, dim=1)
                actions = torch.multinomial(action_probs, num_samples=1).squeeze().cpu().numpy()
        else:
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():  # with torch.inference_mode():  <- issues with torch.compile
                    q_values = q_network(torch.tensor(obs, device=device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminated, truncated, info = envs.step(actions)

        global_step += envs.num_envs
        bar.n = global_step
        bar.refresh()

        if "_episode" in info.keys():
            completed_episodes = info["episode"]["r"][info["_episode"]]
            training_episode_rewards.extend(completed_episodes)

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

                eval_interval = max(1, args.num_iterations // args.n_datapoints_csv) if args.n_datapoints_csv else 1
                do_eval = args.num_iterations == 0 or (update_count % eval_interval == 0) or (
                        update_count == args.num_iterations)
                if do_eval:
                    current_training_time = stop_timer.get_elapsed_time()

                    stop_timer.stop()
                    test_mean, test_median = evaluate_test_performance(q_network, args, device)
                    stop_timer.start()

                    train_mean_reward = np.mean(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0
                    train_median_reward = np.median(training_episode_rewards) if len(
                        training_episode_rewards) > 0 else 0.0

                    log_sit_style_csv(
                        os.path.join(args.output_dir, "sit_format.csv"),
                        loss.item(),  # td_loss (equivalent to action_loss)
                        old_val.mean().item(),  # q_values (equivalent to dist_entropy)
                        0.0,  # value_loss (not applicable for DQN, set to 0)
                        test_mean,  # test_mean
                        test_median,  # test_median
                        train_mean_reward,  # train_mean
                        train_median_reward,  # train_median
                        update_count,  # nupdates
                        global_step,  # total_steps
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

    del replay_buffer
    return envs, q_network, global_step, torch.tensor(obs, device=device)
