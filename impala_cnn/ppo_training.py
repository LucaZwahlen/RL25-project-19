# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import csv
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from impoola.eval import evaluation
from impoola.eval.normalized_score_lists import (progcen_easy_hns,
                                                 progcen_hard_hns, progcen_hns)
from impoola.maker.make_env import make_an_env
from impoola.prune.redo import run_redo
from impoola.train.agents import PPOAgent
from impoola.train.train_ppo_agent import log_metrics_to_csv, train_ppo_agent
from impoola.utils.utils import measure_latency_agent, network_summary


@dataclass
class Args:
    # General Settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    n_episodes_rollout: int = int(2.5e3)
    """the number of episodes to rollout for evaluation"""
    deterministic_rollout: bool = False
    """if toggled, the evaluation will be deterministic"""
    training_eval_ratio: float = 0.1
    """the ratio of training evaluation"""
    normalize_reward: bool = True
    """if toggled, the reward will be normalized"""

    # Algorithm specific arguments
    env_id: str = "fruitbot"
    """the track setting of the environment"""
    distribution_mode: str = "easy"
    """the distribution mode of the environment"""
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 3.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99  # Change from CleanRL, the default is 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # Network specific arguments
    encoder_type: str = "impala"
    """the type of the agent"""
    scale: int = 2
    """the width scale of the network"""
    pruning_type: str = "Baseline"
    """the pruning mode"""
    weight_decay: float = 0.0e-5
    """the weight decay for the optimizer"""
    latent_space_dim: int = 256
    """the latent space dimension"""
    cnn_filters: tuple = (16, 32, 32)
    """the number of filters for each CNN layer"""
    activation: str = 'relu'
    """the activation function of the network"""
    rescale_lr_by_scale: bool = True
    """if toggled, the learning rate will be rescaled by the width scale of the network"""

    # Network improvements

    # ReDo settings
    redo_tau: float = 0.025
    """the tau for the ReDo algorithm"""
    redo_interval: int = 100
    """the interval for the ReDo algorithm (computed in runtime)"""

    # To be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    log_interval: int = 1
    """the logging interval for detailed metrics"""


def save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, checkpoint_name):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(output_dir, f"{checkpoint_name}.pt")
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'global_step': global_step,
        'obs_rms': getattr(envs, 'obs_rms', None),
        'return_rms': getattr(envs, 'return_rms', None),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Create output directory
    output_dir = f"outputs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Pass output_dir to args so train_ppo_agent can access it
    args.output_dir = output_dir

    # Initialize CSV logging with SIT-style headers
    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    sit_format_file = os.path.join(output_dir, "sit_format.csv")

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['global_step', 'metric', 'value'])

    # Initialize SIT-style CSV with header comment
    with open(sit_format_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['losses/action_loss', 'losses/dist_entropy', 'losses/value_loss', 'test/mean_episode_reward', 'test/median_episode_reward',
                        'train/mean_episode_reward', 'train/median_episode_reward', 'train/nupdates', 'train/total_num_steps'])
    global progcen_hns
    if args.distribution_mode == "easy":
        progcen_hns.update(progcen_easy_hns)
    elif args.distribution_mode == "hard":
        progcen_hns.update(progcen_hard_hns)
    else:
        raise ValueError(f"Invalid distribution mode: {args.distribution_mode}")

    print(f"Run name: {run_name} | Batch size: {args.batch_size} | Num iterations: {args.num_iterations}")
    print(f"Outputs will be saved to: {output_dir}")
    print(f"Using SIT-style logging: losses/action_loss, losses/dist_entropy, losses/value_loss, train/test performance every epoch")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == 'cuda':
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # Environment that will be used for training
    envs = make_an_env(args, seed=args.seed,
                       normalize_reward=args.normalize_reward,
                       full_distribution=False)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = PPOAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False

    ).to(device)

    with torch.no_grad():
        example_input = 127 * np.ones((1,) + envs.single_observation_space.shape).astype(
            envs.single_observation_space.dtype)
        example_input = torch.tensor(example_input).to(device)
        agent.get_action_and_value(example_input)

    # print summary of net
    statistics, total_params, m_macs, param_bytes = network_summary(agent, example_input, device)

    # Log initial metrics using the new logging function
    initial_metrics = {
        "total_network_params": total_params,
        "total_network_m_macs": m_macs,
        "total_network_param_bytes": param_bytes,
    }

    log_metrics_to_csv(metrics_file, 0, initial_metrics)

    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,  # default eps=1e-8
        weight_decay=args.weight_decay,
        fused=True
    )

    if args.rescale_lr_by_scale:
        # LR was set for the default scale of 2, so we need to rescale it
        lr_scaling_factor = torch.tensor(args.scale / 2, device=device)
        optimizer.param_groups[0]['lr'].copy_(optimizer.param_groups[0]['lr'] / lr_scaling_factor)
        print(f"Rescaled learning rate to {optimizer.param_groups[0]['lr']}")

    # Save initial checkpoint
    save_checkpoint(agent, optimizer, args, 0, envs, output_dir, "checkpoint_000_initial")

    # TRAINING STEP - now logs SIT-style metrics every epoch
    envs, agent, global_step, b_obs = train_ppo_agent(args, envs, agent, optimizer, device)

    # Save statistics for reward normalization, will be used for fine-tuning on additional levels later
    if args.normalize_reward:
        train_envs_return_norm_mean = envs.return_rms.mean
        train_envs_return_norm_var = envs.return_rms.var
        train_envs_return_norm_count = envs.return_rms.count

    # Save final checkpoint after training
    save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, "checkpoint_100_final")

    envs.close()

    # ANALYSIS OF THE FINAL MODEL
    redo_dict = run_redo(b_obs[:args.minibatch_size], agent, optimizer, args.redo_tau, False, False)

    # Log dormant neuron analysis
    dormant_metrics = {
        "final/zero_fraction": redo_dict['zero_fraction'],
        "final/dormant_fraction": redo_dict['dormant_fraction'],
    }
    for i, (k, v) in enumerate(redo_dict['dormant_neurons_per_layer'].items()):
        dormant_metrics[f"final/dormant_neurons_{i}_{k}"] = v

    log_metrics_to_csv(metrics_file, global_step, dormant_metrics)

    agent = agent.to(device)
    print(f"Zero fraction: {redo_dict['zero_fraction']:.2f} | Dormant fraction: {redo_dict['dormant_fraction']:.2f}")

    # Save config file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # FINAL EVALUATION (additional to the epoch-by-epoch evaluation during training)
    print("Running final detailed evaluation!")
    eval_args = deepcopy(args)

    # EVALUATION TRACK (1): In-distribution generalization only for generalization track
    evaluation.run_training_track(agent, eval_args, global_step)
    # Save checkpoint after evaluation
    save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, "checkpoint_after_training_eval")

    # EVALUATION TRACK (2): Out-of-distribution generalization for full distribution
    evaluation.run_test_track(agent, eval_args, global_step)
    # Save final checkpoint after all evaluations
    save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, "checkpoint_after_test_eval")

    print(f"All training and evaluation complete! Files saved to: {output_dir}")
    print(f"Training metrics logged in SIT format to: {metrics_file}")
