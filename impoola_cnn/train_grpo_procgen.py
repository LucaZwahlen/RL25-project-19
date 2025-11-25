import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import gym
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
from tqdm import trange
from copy import deepcopy
from torch.distributions import Categorical

# --- IMPOOLA & PROCGEN IMPORTS ---
from procgen import ProcgenEnv
from impoola_cnn.impoola.train.nn import encoder_factory, layer_init_orthogonal
from impoola_cnn.impoola.utils.csv_logging import EpisodeQueueCalculator, Logger
from impoola_cnn.impoola.utils.environment_knowledge import TEST_ENV_RANGE
from impoola_cnn.impoola.utils.save_load import save_checkpoint
from impoola_cnn.impoola.utils.utils import get_device
from impoola_cnn.impoola.eval.normalized_score_lists import (
    progcen_easy_hns,
    progcen_hard_hns,
    progcen_hns,
)

# =============================================================================
# 1. CONFIGURATION (ARGS)
# =============================================================================

@dataclass
class Args:
    # --- Experiment Settings ---
    exp_name: str = "grpo_v2_chaser"
    env_id: str = "chaser"
    distribution_mode: str = "easy"
    seed: int = 1
    torch_deterministic: bool = True
    output_dir: str = "outputs"
    
    # --- Training Hyperparameters ---
    total_timesteps: int = int(10e6)
    learning_rate: float = 5e-5  # LOWERED for stability
    num_envs: int = 64         # High count for Chaser
    num_steps: int = 64
    anneal_lr: bool = False
    gamma: float = 0.999
    
    # --- GRPO/PPO Specifics ---
    group_size: int = 8       
    update_epochs: int = 2    
    num_minibatches: int = 8
    kl_coef: float = 0.05      # Penalty for drifting from Ref
    ent_coef: float = 0.05     # Bonus for exploration
    clip_coef: float = 0.2     # SAFETY BELT: Prevents explosion
    max_grad_norm: float = 0.5
    ref_policy_update_freq: int = 50 
    
    # --- Network Architecture ---
    encoder_type: str = "impala"
    scale: int = 2
    latent_space_dim: int = 256
    cnn_filters: Tuple[int, int, int] = (16, 32, 32)
    activation: str = "relu"
    rescale_lr_by_scale: bool = True
    p_augment: float = 0.0
    micro_dropout_p: float = 0.0
    
    # --- Logging & Eval ---
    extensive_logging: bool = True
    is_all_knowing: bool = False
    n_datapoints_csv: int = 500
    normalize_reward: bool = True
    deterministic_rollout: bool = False
    
    # --- Unused/Legacy ---
    move_penalty: float = 0.05
    pruning_type: str = "Baseline"
    weight_decay: float = 0.0
    redo_tau: float = 0.025
    redo_interval: int = 100
    clip_vloss: bool = False
    norm_adv: bool = True
    vf_coef: float = 0.5
    target_kl: Optional[float] = None

# =============================================================================
# 2. CUSTOM ENVIRONMENT WRAPPERS (ROBUST FIX)
# =============================================================================

class VecPyTorch(gym.Wrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.single_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 64, 64), dtype=np.uint8
        )
        self.single_action_space = venv.action_space

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
        else:
            obs = ret
            
        if isinstance(obs, dict) and 'rgb' in obs:
            obs = obs['rgb']

        obs = np.transpose(obs, (0, 3, 1, 2))
        return torch.from_numpy(obs).float().to(self.device) / 255.0

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 5: 
            obs, reward, terminated, truncated, info = ret
            done = np.logical_or(terminated, truncated)
        else: 
            obs, reward, done, info = ret
            
        if isinstance(obs, tuple): obs = obs[0]
        if isinstance(obs, dict) and 'rgb' in obs: obs = obs['rgb']
        
        obs = np.transpose(obs, (0, 3, 1, 2))
        
        obs = torch.from_numpy(obs).float().to(self.device) / 255.0
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        
        return obs, reward, done, False, info

class SyncGroupedProcgenEnv(gym.Wrapper):
    def __init__(self, venv, num_envs, group_size, controller):
        super().__init__(venv)
        self.num_envs = num_envs
        self.group_size = group_size
        self.controller = controller

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
            is_tuple = True
        else:
            obs = ret
            is_tuple = False
            
        try:
            states = self.controller.call_method('get_state')
        except AttributeError:
            states = self.controller.get_state()

        new_states = []
        for i in range(self.num_envs):
            group_id = i // self.group_size
            leader_idx = group_id * self.group_size
            new_states.append(states[leader_idx])
            
        try:
            self.controller.call_method('set_state', new_states)
        except AttributeError:
             self.controller.set_state(new_states)
        
        if isinstance(obs, np.ndarray):
            for i in range(self.num_envs):
                group_id = i // self.group_size
                leader_idx = group_id * self.group_size
                obs[i] = obs[leader_idx]
        elif isinstance(obs, dict) and 'rgb' in obs:
             rgb = obs['rgb']
             for i in range(self.num_envs):
                group_id = i // self.group_size
                leader_idx = group_id * self.group_size
                rgb[i] = rgb[leader_idx]
        
        return (obs, info) if is_tuple else obs

def make_grouped_env(args, device):
    venv = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        num_levels=0, 
        start_level=0,
        distribution_mode=args.distribution_mode,
        rand_seed=args.seed,
    )
    controller = venv.env 
    venv = SyncGroupedProcgenEnv(venv, args.num_envs, args.group_size, controller)
    venv = VecPyTorch(venv, device)
    return venv

# =============================================================================
# 3. MODEL DEFINITION
# =============================================================================

class GRPOAgent(nn.Module):
    def __init__(self, encoder_type, envs, width_scale=1, out_features=256, 
                 cnn_filters=(16, 32, 32), activation='relu', p_augment=0.0, micro_dropout_p=0.0):
        super().__init__()
        
        encoder, out_features = encoder_factory(
            encoder_type=encoder_type,
            envs=envs,
            width_scale=width_scale,
            out_features=out_features,
            cnn_filters=cnn_filters,
            activation=activation,
            use_layer_init_normed=False,
            p_augment=p_augment,
            micro_dropout_p=micro_dropout_p
        )
        self.encoder = encoder
        self.action_head = layer_init_orthogonal(
            nn.Linear(out_features, envs.single_action_space.n), 
            std=0.01
        )

    def forward(self, x):
        logits = self.action_head(self.encoder(x))
        return Categorical(logits=logits)

    def act(self, x):
        pi = self.forward(x)
        a = pi.sample()
        logp = pi.log_prob(a)
        entropy = pi.entropy()
        return a, logp, entropy

# =============================================================================
# 4. TRAINING LOGIC
# =============================================================================

def compute_group_relative_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    N = rewards.shape[0]
    num_full_groups = N // group_size
    truncated_N = num_full_groups * group_size
    
    grouped = rewards[:truncated_N].view(num_full_groups, group_size)
    
    group_means = grouped.mean(dim=1, keepdim=True)
    # Adding a slightly larger epsilon to avoid exploding advantages on flat rewards
    group_stds = grouped.std(dim=1, keepdim=True) + 1e-5 
    
    adv = (grouped - group_means) / group_stds
    return adv.view(-1)

def train_grpo_agent(args, logger, envs, agent, optimizer, device):
    episodeQueueCalculator = EpisodeQueueCalculator(
        'train', args.seed, args.normalize_reward, 100, args.env_id,
        args.num_envs, args.distribution_mode, device
    )

    ref_agent = copy.deepcopy(agent)
    ref_agent.eval()
    for param in ref_agent.parameters(): param.requires_grad = False

    obs_shape = envs.single_observation_space.shape
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    act_buf = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
    rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    logp_old_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    ref_logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs = envs.reset()
    global_step = 0
    cumulative_time = 0.0
    iter_start = time.time()

    for iteration in trange(1, args.num_iterations + 1):
        if args.ref_policy_update_freq > 0 and iteration % args.ref_policy_update_freq == 0:
            ref_agent.load_state_dict(agent.state_dict())

        # --- ROLLOUT ---
        for t in range(args.num_steps):
            obs_buf[t] = next_obs
            global_step += args.num_envs

            with torch.no_grad():
                act, logp_old, _ = agent.act(next_obs)
                ref_pi = ref_agent(next_obs)
                ref_logp = ref_pi.log_prob(act)

            act_buf[t] = act
            logp_old_buf[t] = logp_old
            ref_logp_buf[t] = ref_logp

            next_obs, reward, done, truncated, info = envs.step(act.cpu().numpy())
            rew_buf[t] = reward
            done_buf[t] = done

            episodeQueueCalculator.update(act, rew_buf[t])
            if isinstance(info, list):
                for item in info:
                    if "episode" in item:
                        episodeQueueCalculator.extend({"_episode": [True], "episode": [item["episode"]]})

        # --- RETURNS ---
        returns = torch.zeros_like(rew_buf)
        next_ret = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(args.num_steps)):
            next_ret = rew_buf[t] + args.gamma * next_ret * (1.0 - done_buf[t])
            returns[t] = next_ret

        # --- FLATTEN ---
        B_obs = obs_buf.reshape((-1,) + obs_shape)
        B_act = act_buf.reshape(-1)
        B_ret = returns.reshape(-1)
        B_logp_old = logp_old_buf.reshape(-1)
        B_ref_logp = ref_logp_buf.reshape(-1)

        # --- ADVANTAGE ---
        advantages = compute_group_relative_advantages(B_ret, group_size=args.group_size)

        batch_size = B_obs.shape[0]
        mb_size = int(batch_size // args.num_minibatches)
        indices = torch.arange(batch_size, device=device)

        total_loss, total_ref_kl, total_ent = 0, 0, 0
        updates = 0

        # --- UPDATE ---
        for epoch in range(args.update_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                mb_idx = indices[start : start + mb_size]

                mb_obs = B_obs[mb_idx]
                mb_act = B_act[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_old_logp = B_logp_old[mb_idx]
                mb_ref_logp = B_ref_logp[mb_idx]

                pi = agent(mb_obs)
                logp = pi.log_prob(mb_act)
                entropy = pi.entropy()

                # Ratio
                ratio = (logp - mb_old_logp).exp()
                
                # Ref KL (Purely for logging/penalty, not optimization target)
                with torch.no_grad():
                    # We log the ACTUAL distance from Ref to Current
                    ref_kl = (logp - mb_ref_logp).mean()

                # --- HYBRID GRPO LOSS ---
                # 1. Clipped Surrogate (Safety Belt)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # 2. KL Penalty (Drift control)
                kl_loss = args.kl_coef * (logp - mb_ref_logp).mean() 
                
                # 3. Entropy Bonus
                ent_loss = -args.ent_coef * entropy.mean()

                loss = pg_loss + kl_loss + ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                total_loss += pg_loss.item()
                total_ref_kl += ref_kl.item()
                total_ent += entropy.mean().item()
                updates += 1

        avg_loss = total_loss / updates
        avg_ref_kl = total_ref_kl / updates
        avg_ent = total_ent / updates

        eval_int = max(1, args.num_iterations // args.n_datapoints_csv)
        if iteration % eval_int == 0 or iteration == args.num_iterations:
            cumulative_time += (time.time() - iter_start)
            (tr_rew, tr_med, tr_ticks, tr_steps, tr_succ, tr_spl, tr_lvls, tr_cnt) = episodeQueueCalculator.get_statistics()
            
            print(f"Iter {iteration}: TrainRew {tr_rew:.2f} | KL(Ref) {avg_ref_kl:.4f} | Ent {avg_ent:.4f}")

            logger.log(
                avg_loss, avg_ent, avg_ref_kl, 
                0.0, 0.0, [], 0, 0, 0, 0.0, 0.0,
                tr_rew, tr_med, tr_lvls, tr_cnt, tr_ticks, tr_steps, tr_succ, tr_spl,
                iteration, global_step, cumulative_time,
            )
            iter_start = time.time()

    return envs, agent, global_step

# =============================================================================
# 5. MAIN
# =============================================================================

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Safety
    if args.learning_rate > 1.0: args.learning_rate = 5e-5

    args.batch_size = int(args.num_envs * args.num_steps)
    args.run_name = f"{args.env_id}_GRPOv2_{datetime.now().strftime('%Y%m%d_%H%M')}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = Logger(args)
    global progcen_hns
    progcen_hns.update(progcen_easy_hns if args.distribution_mode == "easy" else progcen_hard_hns)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = get_device()

    print(f"--- GRPO v2 (Clipped) on {device} ---")
    
    envs = make_grouped_env(args, device)
    
    agent = GRPOAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale,
        out_features=args.latent_space_dim,
        cnn_filters=args.cnn_filters,
        activation=args.activation,
    ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    if args.rescale_lr_by_scale:
        optimizer.param_groups[0]["lr"] /= (args.scale / 2)

    args.num_iterations = args.total_timesteps // args.batch_size

    try:
        train_grpo_agent(args, logger, envs, agent, optimizer, device)
        save_checkpoint(agent, optimizer, args, args.total_timesteps, envs, args.output_dir, args.run_name, "final")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        envs.close()
        logger.close()