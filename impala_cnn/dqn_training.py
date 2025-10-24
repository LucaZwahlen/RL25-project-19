# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import csv
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
from impoola.train.agents import DQNAgent
from impoola.train.train_dqn_agent import train_dqn_agent
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
    training_eval_ratio: float = 0.1
    """the ratio of training evaluation"""
    deterministic_rollout: bool = True
    """if toggled, the rollout will be deterministic"""
    normalize_reward: bool = False
    """if toggled, the reward will be normalized"""

    # Algorithm specific arguments
    env_id: str = "fruitbot"
    """the id of the environment"""
    distribution_mode: str = "easy"
    """the distribution mode of the environment"""
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: Optional[float] = None
    """the target network update rate"""
    target_network_frequency: int = 128 * 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.025
    """the ending epsilon for explortion"""
    exploration_fraction: float = 0.1
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 250000
    """timestep to start learning"""
    train_frequency: float = 1
    """the frequency of training"""

    # Additional arguments
    double_dqn: bool = True
    """if toggled, double dqn will be enabled"""
    multi_step: int = 3
    """the number of steps to rollout for multi-step learning"""
    prioritized_replay: bool = True
    """if toggled, prioritized replay will be enabled"""
    softmax_exploration: bool = False
    """if toggled, softmax exploration will be enabled"""
    max_grad_norm: float = 10.0  # 0.5
    """the maximum gradient norm"""
    anneal_lr: bool = False
    """if toggled, the learning rate will be annealed"""

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

    # ReDo settings
    redo_tau: float = 0.025
    """the tau for the ReDo algorithm"""
    redo_interval: int = 2000
    """the interval for the ReDo algorithm (computed in runtime)"""


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


def log_metrics(csv_file, global_step, metrics_dict):
    """Log metrics to CSV file"""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for key, value in metrics_dict.items():
            writer.writerow([global_step, key, value])


if __name__ == "__main__":

    args = tyro.cli(Args)
    num_iterations = args.total_timesteps // args.num_envs // args.train_frequency
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Create output directory
    output_dir = f"outputs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Pass output_dir to args so train_dqn_agent can access it
    args.output_dir = output_dir

    # Initialize CSV logging
    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    sit_format_file = os.path.join(output_dir, "sit_format.csv")

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['global_step', 'metric', 'value'])

    # Initialize SIT-style CSV with header comment
    with open(sit_format_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['losses/td_loss', 'losses/q_values', 'losses/value_loss', 'test/mean_episode_reward', 'test/median_episode_reward',
                        'train/mean_episode_reward', 'train/median_episode_reward', 'train/nupdates', 'train/total_num_steps', 'train/total_time'])
    global progcen_hns
    if args.distribution_mode == "easy":
        progcen_hns.update(progcen_easy_hns)
    elif args.distribution_mode == "hard":
        progcen_hns.update(progcen_hard_hns)
    else:
        raise ValueError(f"Invalid distribution mode: {args.distribution_mode}")

    print(f"Run name: {run_name} | Batch size: {args.batch_size} | Num iterations: {num_iterations}")
    print(f"Outputs will be saved to: {output_dir}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # Environment that will be used for training
    envs = make_an_env(args, seed=args.seed,
                       normalize_reward=args.normalize_reward,
                       full_distribution=False)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Network creation
    q_network = DQNAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,

    ).to(device)

    target_network = DQNAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,

    ).to(device)

    with torch.no_grad():
        example_input = 127 * np.ones((1,) + envs.single_observation_space.shape).astype(
            envs.single_observation_space.dtype)
        example_input = torch.tensor(example_input).to(device)
        q_network(example_input)
        target_network(example_input)

    # print summary of net
    statistics, total_params, m_macs, param_bytes = network_summary(q_network, example_input, device)

    # Log initial metrics
    initial_metrics = {
        "charts/total_network_params": total_params,
        "charts/total_network_m_macs": m_macs,
        "charts/total_network_param_bytes": param_bytes,
    }

    log_metrics(metrics_file, 0, initial_metrics)

    optimizer = optim.Adam(
        q_network.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,  # default eps=1e-8
        weight_decay=args.weight_decay,
        fused=True
    )

    if args.rescale_lr_by_scale:
        # LR was set for the default scale of 2, so we need to rescale it
        lr_scaling_factor = torch.tensor(args.scale / 2, device=device)
        optimizer.param_groups[0]['lr'].copy_(optimizer.param_groups[0]['lr'] / lr_scaling_factor)

    # copy the q_network
    target_network.load_state_dict(q_network.state_dict())

    # TRAINING STEP
    envs, q_network, global_step, b_obs = train_dqn_agent(args, envs, (q_network, target_network), optimizer, device)

    # Save statistics for reward normalization, will be used for fine-tuning on additional levels later
    if args.normalize_reward:
        train_envs_return_norm_mean = envs.return_rms.mean
        train_envs_return_norm_var = envs.return_rms.var
        train_envs_return_norm_count = envs.return_rms.count

    # Done!
    envs.close()
    agent = q_network

    # ANALYSIS OF THE FINAL MODEL
    save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, 'checkpoint_final')

    # Check how much GPU memory is used. If less than 5 GB are available, run the redo algorithm on the CPU
    if torch.cuda.get_device_properties(0).total_memory < 5e9:
        agent = agent.to('cpu')
        b_obs = b_obs.to('cpu')
        print("Running ReDo on CPU")

    redo_dict = run_redo(b_obs[:32], agent, optimizer, args.redo_tau, False, False)

    # Log dormant neuron analysis
    dormant_metrics = {
        "zero_fraction": redo_dict['zero_fraction'],
        "dormant_fraction": redo_dict['dormant_fraction'],
    }
    for i, (k, v) in enumerate(redo_dict['dormant_neurons_per_layer'].items()):
        dormant_metrics[f"dormant_neurons_{i}_{k}"] = v

    log_metrics(metrics_file, global_step, dormant_metrics)

    agent = agent.to(device)
    print(f"Zero fraction: {redo_dict['zero_fraction']:.2f} | Dormant fraction: {redo_dict['dormant_fraction']:.2f}")

    # EVALUATION
    print("Running evaluation!")
    eval_args = deepcopy(args)

    # EVALUATION TRACK (1): In-distribution generalization only for generalization track
    evaluation.run_training_track(agent, eval_args, global_step)

    # EVALUATION TRACK (2): Out-of-distribution generalization for full distribution
    if args.env_track_setting == "generalization":
        evaluation.run_test_track(agent, eval_args, global_step)
