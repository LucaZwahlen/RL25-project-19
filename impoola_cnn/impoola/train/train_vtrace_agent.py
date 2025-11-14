import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from impoola_cnn.impoola.train.vtrace_criterion import compute_vtrace_targets
from impoola_cnn.impoola.utils.DRAC import DRACTransformChaserFruitbot, remap_logprobs_for_flip
from impoola_cnn.impoola.utils.csv_logging import (EpisodeQueueCalculator, Logger)
from impoola_cnn.impoola.utils.evaluate_test_performance import evaluate_test_performance


def train_vtrace_agent(args, logger: Logger, envs, agent, optimizer, device):
    try:
        drac_transform = DRACTransformChaserFruitbot().to(device)

        T = args.unroll_length
        N = args.num_envs

        obs = torch.zeros((T, N, 3, 64, 64), device=device, dtype=torch.uint8)
        actions = torch.zeros((T, N), device=device, dtype=torch.long)
        rewards = torch.zeros((T, N), device=device, dtype=torch.float32)
        dones = torch.zeros((T, N), device=device, dtype=torch.bool)
        values = torch.zeros((T, N), device=device, dtype=torch.float32)
        behavior_logits = torch.zeros((T, N, envs.single_action_space.n), device=device, dtype=torch.float32)

        gamma = torch.tensor(args.gamma, device=device)
        ent_coef = torch.tensor(args.ent_coef, device=device)
        vf_coef = torch.tensor(args.vf_coef, device=device)
        rho_bar = torch.tensor(args.vtrace_rho_bar, device=device)
        c_bar = torch.tensor(args.vtrace_c_bar, device=device)

        episodeQueueCalculator = EpisodeQueueCalculator('train', args.seed, args.normalize_reward, 100, args.env_id, N,
                                                        args.distribution_mode, device)

        global_step = 0

        cumulative_training_time = 0.0
        iteration_start_time = time.time()

        next_obs_np, _ = envs.reset()
        next_obs = torch.from_numpy(next_obs_np).to(device, non_blocking=True).to(torch.uint8)

        for iteration in trange(1, args.num_iterations + 1):

            for step in range(0, T):
                global_step += N
                obs[step] = next_obs

                with torch.no_grad():
                    action, _, _, value, pi_logits = agent.get_action_and_value(next_obs)
                    actions[step] = action
                    values[step] = value.flatten()
                    behavior_logits[step] = pi_logits

                np_actions = actions[step].detach().cpu().numpy()
                next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(np_actions)

                rewards[step] = torch.as_tensor(reward_np, device=device, dtype=torch.float32)
                dones[step] = torch.as_tensor(np.logical_or(terminated_np, truncated_np), device=device,
                                              dtype=torch.bool)
                next_obs = torch.from_numpy(next_obs_np).to(device, non_blocking=True).to(torch.uint8)

                episodeQueueCalculator.update(actions[step], rewards[step])

                if "_episode" in info.keys():
                    episodeQueueCalculator.extend(info)

            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            T_, N_ = T, N

            with torch.no_grad():
                bootstrap_value = agent.get_pi_and_value(next_obs)[1].squeeze(-1)

            flat_obs = b_obs
            target_pi, target_values_flat = agent.get_pi_and_value(flat_obs)
            target_logits_flat = target_pi.logits
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
                actions=actions.reshape(T * N)
            )
            # pg_adv = (pg_adv - pg_adv.mean()) / (pg_adv.std() + 1e-8)

            flat_target_pi = torch.distributions.Categorical(logits=target_logits_flat)
            logp = flat_target_pi.log_prob(actions.reshape(T * N)).view(T, N)
            entropy = flat_target_pi.entropy().view(T, N)

            policy_loss = -(pg_adv.detach() * logp).mean()

            target_values = target_values_flat.reshape(T_, N_)
            vs = vs.reshape(T_, N_).to(dtype=target_values_flat.dtype)
            value_loss = 0.5 * (target_values - vs.detach()).pow(2).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

            if args.drac_lambda > 0.0:
                flat_actions = actions.reshape(T_ * N_)
                logits_full = target_logits_flat
                val_full = target_values_flat.squeeze(-1)
                obs_t = drac_transform(b_obs.reshape(T_ * N_, *envs.single_observation_space.shape))
                pi_t, values_t = agent.get_pi_and_value(obs_t)

                values_t = values_t.squeeze(-1)
                drac_value_loss = (val_full.detach() - values_t).pow(2).mean()

                dist_clean = torch.distributions.Categorical(logits=logits_full)
                dist_flip = torch.distributions.Categorical(logits=pi_t.logits)

                logp_clean = dist_clean.log_prob(flat_actions)
                logp_flip = remap_logprobs_for_flip(
                    dist_flip,
                    flat_actions
                )
                drac_policy_loss = (logp_clean.detach() - logp_flip).pow(2).mean()
                loss = loss + args.drac_lambda * (drac_value_loss + drac_policy_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            eval_interval = max(1, args.num_iterations // args.n_datapoints_csv) if args.n_datapoints_csv else 1
            do_eval = args.num_iterations == 0 or (iteration % eval_interval == 0) or (iteration == args.num_iterations)
            if do_eval:
                avg_policy_loss = policy_loss.item()
                avg_value_loss = value_loss.item()
                avg_entropy_loss = entropy_loss.item()

                iteration_end_time = time.time()
                cumulative_training_time += (iteration_end_time - iteration_start_time)

                simple, detailed = evaluate_test_performance(agent, args, device)

                test_mean_reward, test_median_reward, test_ticks, test_steps, test_success, test_spl, test_levels, test_count = simple
                test_rewards, test_num_ticks, test_num_steps, test_is_success, test_spl_terms, _ = detailed

                train_mean_reward, train_median_reward, train_ticks, train_steps, train_success, train_spl, train_levels, train_count = episodeQueueCalculator.get_statistics()

                logger.log(
                    avg_policy_loss,
                    avg_entropy_loss,
                    avg_value_loss,
                    test_mean_reward,
                    test_median_reward,
                    test_levels,
                    test_count,
                    test_ticks,
                    test_steps,
                    test_success,
                    test_spl,
                    train_mean_reward,
                    train_median_reward,
                    train_levels,
                    train_count,
                    train_ticks,
                    train_steps,
                    train_success,
                    train_spl,
                    iteration,
                    global_step,
                    cumulative_training_time
                )

                if args.extensive_logging:
                    train_rewards, train_num_ticks, train_num_steps, train_is_success, train_spl_terms, _ = episodeQueueCalculator.get_raw_counts()

                    logger.log_extensive(
                        avg_policy_loss,
                        avg_entropy_loss,
                        avg_value_loss,
                        test_rewards,
                        test_num_ticks,
                        test_num_steps,
                        test_is_success,
                        test_spl_terms,
                        train_rewards,
                        train_num_ticks,
                        train_num_steps,
                        train_is_success,
                        train_spl_terms,
                        iteration,
                        global_step,
                        cumulative_training_time
                    )

                iteration_start_time = time.time()


    except KeyboardInterrupt:
        pass

    finally:
        return envs, agent, global_step, b_obs
