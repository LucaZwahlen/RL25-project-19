# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from impoola_cnn.impoola.utils.DRAC import DRACTransformChaserFruitbot, remap_logprobs_for_flip
from impoola_cnn.impoola.utils.csv_logging import (EpisodeQueueCalculator,
                                                   Logger)
from impoola_cnn.impoola.utils.evaluate_test_performance import \
    evaluate_test_performance
class RNDModel(nn.Module):
    """Simple RND module with a target (fixed) and predictor (trainable).
    Uses small conv encoder + adaptive pooling to support arbitrary image sizes.
    """
    def __init__(self, obs_shape, rnd_output_size=128):
        super().__init__()
        c, h, w = obs_shape
        self.predictor = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, rnd_output_size)
        )
        self.target = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, rnd_output_size)
        )
        # Freeze target parameters
        for p in self.target.parameters():
            p.requires_grad = False


def train_ppo_agent(args, logger: Logger, envs, agent, optimizer, device):
    """ Train the PPO agent """

    # Track training episode statistics
    episodeQueueCalculator = EpisodeQueueCalculator('all-knowing' if args.is_all_knowing else 'train', args.seed, args.normalize_reward,
                                                    100, args.env_id, args.num_envs, args.distribution_mode, device)

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

    drac_transform = DRACTransformChaserFruitbot().to(device)

    from impoola_cnn.impoola.train.ppo_criterion import ppo_gae, ppo_loss

    # TRY NOT TO MODIFY: start the game
    global_step = 0

    # Initialize cumulative training timer (excluding evaluation time)
    cumulative_training_time = 0.0
    iteration_start_time = time.time()

    next_obs, _ = envs.reset()

    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    #RND stuff initialization
    if args.use_rnd:
        # observations are already normalized according to your confirmation
        rnd = RNDModel(envs.single_observation_space.shape, args.rnd_output_size).to(device)
        rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=args.rnd_lr)
    else:
        rnd = None
        rnd_optimizer = None
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

            # Compute RND intrinsic reward and add to environment reward
            if args.use_rnd:
                # next_obs: (num_envs, C, H, W)
                with torch.no_grad():
                    target_feat = rnd.target(next_obs)
                pred_feat = rnd.predictor(next_obs)
                intrinsic_reward = (target_feat - pred_feat).pow(2).mean(dim=1)
                # rewards[step] shape is (num_envs,)
                rewards[step] += args.rnd_coef * intrinsic_reward


            episodeQueueCalculator.update(action, rewards[step])

            # Collect training episode rewards
            if "_episode" in info.keys():
                episodeQueueCalculator.extend(info)

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
        #train the RND predictor network
        if args.use_rnd:
            # b_obs is shape (batch_size, C, H, W) and already on device
            rnd_optimizer.zero_grad()
            pred_feat = rnd.predictor(b_obs)
            with torch.no_grad():
                target_feat = rnd.target(b_obs)
            rnd_loss = (pred_feat - target_feat).pow(2).mean()
            rnd_loss.backward()
            rnd_optimizer.step()
        # Optimizing the policy and value network
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0

        T_ = args.num_steps
        N_ = args.num_envs

        target_logits_flat = logits.reshape(T_ * N_, -1)
        target_values_flat = values.reshape(T_ * N_)

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

                if args.drac_lambda > 0.0:
                    flat_actions = b_actions[mb_inds].reshape(-1)
                    logits_full = target_logits_flat[mb_inds]
                    val_full = target_values_flat[mb_inds]

                    obs_t = drac_transform(b_obs[mb_inds].reshape(len(mb_inds), *envs.single_observation_space.shape))
                    pi_t, values_t = agent.get_pi_and_value(obs_t)
                    values_t = values_t.squeeze(-1) if values_t.ndim > 1 else values_t

                    drac_value_loss = (val_full.detach() - values_t).pow(2).mean()

                    dist_clean = torch.distributions.Categorical(logits=logits_full)
                    dist_flip = torch.distributions.Categorical(logits=pi_t.logits if hasattr(pi_t, "logits") else pi_t)

                    logp_clean = dist_clean.log_prob(flat_actions)
                    logp_flip = remap_logprobs_for_flip(dist_flip, flat_actions)

                    drac_policy_loss = (logp_clean.detach() - logp_flip).pow(2).mean()
                    loss = loss + args.drac_lambda * (drac_value_loss + drac_policy_loss)

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
            avg_policy_loss = total_policy_loss / n_updates
            avg_value_loss = total_value_loss / n_updates
            avg_entropy_loss = total_entropy_loss / n_updates

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
