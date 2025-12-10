import copy
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

# Ensure these are imported from your project structure
from impoola_cnn.impoola.utils.csv_logging import EpisodeQueueCalculator
from impoola_cnn.impoola.utils.evaluate_test_performance import (
    evaluate_test_performance_grpo,
)


def compute_group_relative_advantages(
    rewards: torch.Tensor, group_size: int
) -> torch.Tensor:
    """
    Calculates advantage by normalizing returns within a group.
    """
    N = rewards.shape[0]
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    # If batch is smaller than group_size, just subtract global mean
    if N <= group_size:
        group_mean = rewards.mean()
        group_std = rewards.std() + 1e-8
        return (rewards - group_mean) / group_std

    num_full_groups = N // group_size
    full_part = rewards[: num_full_groups * group_size]
    tail_part = rewards[num_full_groups * group_size :]

    grouped = full_part.view(num_full_groups, group_size)
    group_means = grouped.mean(dim=1, keepdim=True)
    group_stds = grouped.std(dim=1, keepdim=True) + 1e-8

    full_adv = (grouped - group_means) / group_stds
    full_adv = full_adv.view(-1)

    if tail_part.numel() > 0:
        tail_mean = tail_part.mean()
        tail_std = tail_part.std() + 1e-8
        tail_adv = (tail_part - tail_mean) / tail_std
        advantages = torch.cat([full_adv, tail_adv], dim=0)
    else:
        advantages = full_adv

    return advantages


def train_grpo_agent(args, logger, envs, agent, optimizer, device):
    episodeQueueCalculator = EpisodeQueueCalculator(
        "all-knowing" if args.is_all_knowing else "train",
        args.seed,
        args.normalize_reward,
        100,
        args.env_id,
        args.num_envs,
        args.distribution_mode,
        device,
    )

    # --- 1. SETUP REFERENCE MODEL (CRITICAL FOR GRPO) ---
    # We need a frozen copy of the agent to calculate KL divergence
    ref_agent = copy.deepcopy(agent).to(device)
    ref_agent.eval()
    for param in ref_agent.parameters():
        param.requires_grad = False

    # Rollout buffers
    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    act_buf = torch.zeros(
        (args.num_steps, args.num_envs), device=device, dtype=torch.long
    )
    rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Refs for KL calculation buffer
    ref_logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = 0
    cumulative_training_time = 0.0
    iteration_start_time = time.time()

    for iteration in trange(1, args.num_iterations + 1):

        # Update Reference Policy periodically
        if (
            args.ref_policy_update_freq > 0
            and iteration % args.ref_policy_update_freq == 0
        ):
            ref_agent.load_state_dict(agent.state_dict())

        # ---------------- ROLLOUT ----------------
        for t in range(args.num_steps):
            obs_buf[t] = next_obs
            global_step += args.num_envs

            with torch.no_grad():
                # Get Current Policy Action
                act, logp, entropy = agent.act(next_obs)

                # Get Reference Policy LogProb (For KL calculation later)
                # We strictly need the prob of the action chosen by the *current* agent
                # calculated by the *ref* agent.
                ref_pi = ref_agent(next_obs)
                ref_logp = ref_pi.log_prob(act)

            act_buf[t] = act
            ref_logp_buf[t] = ref_logp  # Store ref logp

            next_obs, reward, terminated, truncated, info = envs.step(act.cpu().numpy())
            next_obs = torch.tensor(next_obs, device=device)

            rew_buf[t] = torch.tensor(reward, device=device, dtype=torch.float32)
            done = np.logical_or(terminated, truncated)
            done_buf[t] = torch.tensor(done, device=device, dtype=torch.float32)
            next_done = done_buf[t]

            episodeQueueCalculator.update(act, rew_buf[t])
            if "_episode" in info.keys():
                episodeQueueCalculator.extend(info)

        # ---------------- CALCULATE RETURNS ----------------
        returns = torch.zeros_like(rew_buf)
        next_ret = torch.zeros(args.num_envs, device=device)

        for t in reversed(range(args.num_steps)):
            next_ret = rew_buf[t] + args.gamma * next_ret * (1.0 - done_buf[t])
            returns[t] = next_ret

        # ---------------- FLATTEN BATCH ----------------
        B_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        B_act = act_buf.reshape(-1)
        B_ret = returns.reshape(-1)
        B_ref_logp = ref_logp_buf.reshape(-1)

        batch_size = B_obs.shape[0]

        # ---------------- ADVANTAGE (GROUP-RELATIVE) ----------------
        advantages = compute_group_relative_advantages(
            B_ret,
            group_size=args.num_envs,
        )

        # ---------------- UPDATE θ ----------------
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0

        indices = torch.randperm(batch_size, device=device)

        for start in range(0, batch_size, args.minibatch_size):
            end = min(start + args.minibatch_size, batch_size)
            mb_idx = indices[start:end]

            mb_obs = B_obs[mb_idx]
            mb_act = B_act[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ref_logp = B_ref_logp[mb_idx]

            pi = agent(mb_obs)
            logp = pi.log_prob(mb_act)
            entropy = pi.entropy()

            # --- KL DIVERGENCE (Approx) ---
            # KL(pi || ref) = log(pi) - log(ref) (sampled approximation)
            # We want to punish divergence from reference
            # http://joschu.net/blog/kl-approx.html
            with torch.no_grad():
                # simple approximation: logp - ref_logp
                # Note: ref_logp is fixed from rollout, logp is differentiable
                approx_kl = logp - mb_ref_logp

            # GRPO Loss:
            # L = - (Adv * logp) + (beta * KL) - (ent_coef * entropy)

            # 1. Policy Gradient
            pg_loss = -(mb_adv.detach() * logp).mean()

            # 2. KL Penalty (Critical for GRPO stability)
            kl_loss = args.kl_coef * approx_kl.mean()

            # 3. Entropy Bonus
            entropy_bonus = args.ent_coef * entropy.mean()

            loss = pg_loss + kl_loss - entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            total_policy_loss += pg_loss.item()
            total_entropy += entropy.mean().item()
            total_kl += approx_kl.mean().item()
            num_updates += 1

        avg_policy_loss = total_policy_loss / num_updates
        avg_entropy_loss = total_entropy / num_updates  # logged as positive usually
        avg_kl = total_kl / num_updates

        # ---------------- LOGGING ----------------
        eval_interval = max(1, args.num_iterations // args.n_datapoints_csv)
        if iteration % eval_interval == 0 or iteration == args.num_iterations:
            iteration_end_time = time.time()
            cumulative_training_time += iteration_end_time - iteration_start_time

            simple, detailed = evaluate_test_performance_grpo(agent, args, device)
            (
                test_mean_reward,
                test_median_reward,
                test_ticks,
                test_steps,
                test_success,
                test_spl,
                test_levels,
                test_count,
            ) = simple

            (
                train_mean_reward,
                train_median_reward,
                train_ticks,
                train_steps,
                train_success,
                train_spl,
                train_levels,
                train_count,
            ) = episodeQueueCalculator.get_statistics()

            # Added KL logging
            print(
                f"Iter {iteration}: Rew {train_mean_reward:.2f} | KL {avg_kl:.4f} | Ent {avg_entropy_loss:.4f}"
            )

            logger.log(
                avg_policy_loss,
                avg_entropy_loss,  # Log raw entropy, not negative
                avg_kl,  # Replaced value_loss with KL since we have no critic
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
                cumulative_training_time,
            )

            iteration_start_time = time.time()

    return envs, agent, global_step, B_obs
