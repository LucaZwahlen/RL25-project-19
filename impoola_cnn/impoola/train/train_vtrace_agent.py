import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from impoola_cnn.impoola.train.vtrace_criterion import compute_vtrace_targets
from impoola_cnn.impoola.utils.DRAC import DRACTransformChaserFruitbot, remap_logprobs_for_flip
from impoola_cnn.impoola.utils.csv_logging import (EpisodeQueueCalculator,
                                                   Logger)
from impoola_cnn.impoola.utils.evaluate_test_performance import \
    evaluate_test_performance


def train_vtrace_agent(args, logger: Logger, envs, agent, optimizer, device):
    drac_transform = DRACTransformChaserFruitbot(
        hflip=args.drac_hflip,
        vflip=args.drac_vflip
    ).to(device)

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

    episodeQueueCalculator = EpisodeQueueCalculator('train', args.seed, args.normalize_reward, 100, args.env_id, N, args.distribution_mode, device)

    global_step = 0
    cumulative_training_time = 0.0
    iteration_start_time = time.time()

    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, device=device)
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
            episodeQueueCalculator.update(action, rewards[step])
            if "_episode" in info.keys():
                episodeQueueCalculator.extend(info)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        T_, N_ = T, N

        with torch.no_grad():
            bootstrap_value = agent.get_pi_and_value(next_obs)[1].squeeze(-1)

        target_pi, _ = agent.get_pi_and_value(b_obs)
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
            actions=actions.reshape(T_ * N_)
        )

        flat_actions = actions.reshape(T_ * N_)
        flat_pg_adv = pg_adv.reshape(T_ * N_)
        flat_vs = vs.reshape(T_ * N_)
        flat_behavior_logits = behavior_logits.reshape(T_ * N_, -1)

        B = T_ * N_

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_updates = 0

        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(B, device=device)
            for start in range(0, B, args.minibatch_size):
                end = min(start +args.minibatch_size, B)
                mb_inds = b_inds[start:end]

                pi_mb, val_mb = agent.get_pi_and_value(b_obs[mb_inds])
                logits_mb = pi_mb.logits
                dist_mb = torch.distributions.Categorical(logits=logits_mb)

                logp_mb = dist_mb.log_prob(flat_actions[mb_inds])
                entropy_mb = dist_mb.entropy()

                pg_adv_used_mb = flat_pg_adv[mb_inds].detach()
                if args.norm_adv:
                    pg_adv_used_mb = (pg_adv_used_mb - pg_adv_used_mb.mean()) / (pg_adv_used_mb.std() + 1e-8)

                policy_loss_mb = -(pg_adv_used_mb * logp_mb).mean()

                vs_tgt_mb = flat_vs[mb_inds].detach()
                value_error_mb = (val_mb.squeeze(-1) - vs_tgt_mb)
                value_loss_mb = 0.5 * value_error_mb.pow(2).mean()
                if args.clip_vloss:
                    unclipped = val_mb.squeeze(-1)
                    clipped_value = unclipped + torch.clamp(vs_tgt_mb - unclipped, -args.clip_coef, args.clip_coef)
                    clipped_value_error = (clipped_value - vs_tgt_mb)
                    clipped_v_loss = 0.5 * clipped_value_error.pow(2).mean()
                    value_loss_mb = torch.max(value_loss_mb, clipped_v_loss)

                entropy_loss_mb = entropy_mb.mean()

                loss_mb = policy_loss_mb + vf_coef * value_loss_mb - ent_coef * entropy_loss_mb

                if args.kl_coef > 0.0:
                    dist_beh_mb = torch.distributions.Categorical(logits=flat_behavior_logits[mb_inds])
                    kl_mb = torch.distributions.kl_divergence(
                        torch.distributions.Categorical(logits=logits_mb),
                        dist_beh_mb
                    ).mean()
                    loss_mb = loss_mb + args.kl_coef * kl_mb

                if args.drac_lambda_v > 0.0 or args.drac_lambda_pi > 0.0:
                    drac_value_loss_mb = torch.tensor(0.0, device=device)
                    drac_policy_loss_mb = torch.tensor(0.0, device=device)
                    obs_t_mb = drac_transform(b_obs[mb_inds])
                    if args.drac_lambda_v > 0.0:
                        _, values_t_mb = agent.get_pi_and_value(obs_t_mb)
                        drac_value_loss_mb = (val_mb.detach() - values_t_mb).pow(2).mean()
                    if args.drac_lambda_pi > 0.0:
                        pi_t_mb, _ = agent.get_pi_and_value(obs_t_mb)
                        dist_clean_mb = torch.distributions.Categorical(logits=logits_mb)
                        dist_flip_mb = torch.distributions.Categorical(logits=pi_t_mb.logits)
                        logp_clean_mb = dist_clean_mb.log_prob(flat_actions[mb_inds])
                        logp_flip_mb = remap_logprobs_for_flip(
                            dist_flip_mb,
                            flat_actions[mb_inds],
                            hflip=args.drac_hflip,
                            vflip=args.drac_vflip
                        )
                        drac_policy_loss_mb = (logp_clean_mb.detach() - logp_flip_mb).pow(2).mean()
                    loss_mb = loss_mb + args.drac_lambda_v * drac_value_loss_mb + args.drac_lambda_pi * drac_policy_loss_mb

                optimizer.zero_grad(set_to_none=True)
                loss_mb.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss_mb.item()
                total_value_loss += value_loss_mb.item()
                total_entropy_loss += entropy_loss_mb.item()
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

    return envs, agent, global_step, b_obs
