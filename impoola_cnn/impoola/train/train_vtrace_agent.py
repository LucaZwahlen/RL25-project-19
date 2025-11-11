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
        drac_transform = DRACTransformChaserFruitbot(hflip=args.drac_hflip, vflip=args.drac_vflip).to(device)
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
        learning_rate = optimizer.param_groups[0]["lr"]
        max_grad_norm = torch.tensor(args.max_grad_norm, device=device)
        episodeQueueCalculator = EpisodeQueueCalculator('train', args.seed, args.normalize_reward, 100, args.env_id, N,
                                                        args.distribution_mode, device)
        global_step = 0
        cumulative_training_time = 0.0
        iteration_start_time = time.time()
        next_obs, _ = envs.reset()
        next_obs = torch.as_tensor(next_obs, device=device)
        next_done = torch.zeros(N, device=device, dtype=torch.bool)

        for iteration in trange(1, args.num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * learning_rate

            for step in range(0, T):
                global_step += N
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, _, ent, value, pi_logits = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                    behavior_logits[step] = pi_logits
                actions[step] = action
                next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(action.cpu().numpy())
                next_done_np = np.logical_or(terminated_np, truncated_np)
                rewards[step] = torch.as_tensor(reward_np, device=device).view(-1)
                next_obs = torch.as_tensor(next_obs_np, device=device)
                next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.bool)
                episodeQueueCalculator.update(action, rewards[step])
                if "_episode" in info.keys():
                    episodeQueueCalculator.extend(info)

            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            T_, N_ = T, N

            with torch.no_grad():
                bootstrap_value = agent.get_pi_and_value(next_obs)[1].squeeze(-1)

            target_pi, _ = agent.get_pi_and_value(b_obs)
            target_logits = target_pi.logits.view(T_, N_, -1)

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

            flat_actions = actions.view(T_ * N_)
            flat_pg_adv = pg_adv.view(T_ * N_)
            flat_vs = vs.view(T_ * N_)
            flat_behavior_logits = behavior_logits.view(T_ * N_, -1)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy_loss = 0.0
            n_updates = 0

            for _ in range(args.update_epochs):
                pi_full, val_full = agent.get_pi_and_value(b_obs)
                logits_full = pi_full.logits
                dist_full = torch.distributions.Categorical(logits=logits_full)
                logp_full = dist_full.log_prob(flat_actions)
                entropy_full = dist_full.entropy()

                pg_adv_used = flat_pg_adv.detach()
                if args.norm_adv:
                    pg_adv_used = (pg_adv_used - pg_adv_used.mean()) / (pg_adv_used.std() + 1e-8)

                policy_loss = -(pg_adv_used * logp_full).mean()

                vs_tgt = flat_vs.detach()
                value_error = (val_full.squeeze(-1) - vs_tgt)
                value_loss = 0.5 * value_error.pow(2).mean()
                if args.clip_vloss:
                    unclipped = val_full.squeeze(-1)
                    clipped_value = unclipped + torch.clamp(vs_tgt - unclipped, -args.clip_coef, args.clip_coef)
                    clipped_value_error = (clipped_value - vs_tgt)
                    clipped_v_loss = 0.5 * clipped_value_error.pow(2).mean()
                    value_loss = torch.maximum(value_loss, clipped_v_loss)

                entropy_loss = entropy_full.mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                if args.kl_coef > 0.0:
                    dist_beh = torch.distributions.Categorical(logits=flat_behavior_logits)
                    kl_full = torch.distributions.kl_divergence(dist_full, dist_beh).mean()
                    loss = loss + args.kl_coef * kl_full

                if args.drac_lambda_v > 0.0 or args.drac_lambda_pi > 0.0:
                    drac_value_loss = torch.tensor(0.0, device=device)
                    drac_policy_loss = torch.tensor(0.0, device=device)
                    obs_t = drac_transform(b_obs)
                    if args.drac_lambda_v > 0.0:
                        _, values_t = agent.get_pi_and_value(obs_t)
                        drac_value_loss = (val_full.detach() - values_t).pow(2).mean()
                    if args.drac_lambda_pi > 0.0:
                        pi_t, _ = agent.get_pi_and_value(obs_t)
                        dist_clean = torch.distributions.Categorical(logits=logits_full)
                        dist_flip = torch.distributions.Categorical(logits=pi_t.logits)
                        logp_clean = dist_clean.log_prob(flat_actions)
                        logp_flip = remap_logprobs_for_flip(
                            dist_flip,
                            flat_actions,
                            hflip=args.drac_hflip,
                            vflip=args.drac_vflip
                        )
                        drac_policy_loss = (logp_clean.detach() - logp_flip).pow(2).mean()
                    loss = loss + args.drac_lambda_v * drac_value_loss + args.drac_lambda_pi * drac_policy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1

            eval_interval = max(1, args.num_iterations // args.n_datapoints_csv) if args.n_datapoints_csv else 1
            do_eval = args.num_iterations == 0 or (iteration % eval_interval == 0) or (iteration == args.num_iterations)
            if do_eval:
                avg_policy_loss = total_policy_loss / max(1, n_updates)
                avg_value_loss = total_value_loss / max(1, n_updates)
                avg_entropy_loss = total_entropy_loss / max(1, n_updates)
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
