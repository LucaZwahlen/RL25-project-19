# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import gym
import numpy as np
import torch
import torch.optim as optim
import tyro

from impoola_cnn.impoola.eval.normalized_score_lists import (progcen_easy_hns,
                                                             progcen_hard_hns,
                                                             progcen_hns)
from impoola_cnn.impoola.maker.make_env import make_an_env
from impoola_cnn.impoola.train.agents import DQNAgent
from impoola_cnn.impoola.train.train_dqn_agent import train_dqn_agent
from impoola_cnn.impoola.utils.csv_logging import Logger
from impoola_cnn.impoola.utils.save_load import save_checkpoint
from impoola_cnn.impoola.utils.utils import get_device


@dataclass
class Args:

    extensive_logging: bool = True  # whether to log detailed per-episode data

    # General Settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    n_episodes_rollout: int = int(2.5e3)
    training_eval_ratio: float = 0.1
    deterministic_rollout: bool = True
    normalize_reward: bool = False

    # Algorithm specific arguments
    env_id: str = "fruitbot"
    distribution_mode: str = "easy"
    total_timesteps: int = int(25e6)
    learning_rate: float = 1e-4
    num_envs: int = 128
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: Optional[float] = None
    target_network_frequency: int = 128 * 500
    batch_size: int = 512
    start_e: float = 1.0
    end_e: float = 0.025
    exploration_fraction: float = 0.1
    learning_starts: int = 250000
    train_frequency: float = 1

    # Additional arguments
    multi_step: int = 3
    softmax_exploration: bool = False
    max_grad_norm: float = 10.0  # 0.5
    anneal_lr: bool = False
    prioritized_replay: bool = True

    # Network specific arguments
    encoder_type: str = "impala"
    scale: int = 2
    pruning_type: str = "Baseline"
    weight_decay: float = 0.0e-5
    latent_space_dim: int = 256
    cnn_filters: tuple = (16, 32, 32)
    activation: str = 'relu'
    rescale_lr_by_scale: bool = True

    # ReDo settings
    redo_tau: float = 0.025
    redo_interval: int = 2000

    p_augment: float = 0.0
    micro_dropout_p: float = 0.0

    num_iterations = total_timesteps // num_envs // train_frequency
    run_name = f"{env_id}__{exp_name}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir: str = os.path.join("outputs", run_name)

    n_datapoints_csv: int = 500


if __name__ == "__main__":

    args = tyro.cli(Args)
    logger = Logger(args)

    global progcen_hns
    if args.distribution_mode == "easy":
        progcen_hns.update(progcen_easy_hns)
    elif args.distribution_mode == "hard":
        progcen_hns.update(progcen_hard_hns)
    else:
        raise ValueError(f"Invalid distribution mode: {args.distribution_mode}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = get_device()

    line = "=" * 72
    sub = "-" * 72
    title = "TRAINING RUN CONFIGURATION"

    print(line)
    print(f"{title:^72}")
    print(sub)
    print(f"Run name       : {args.run_name}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Num iterations : {args.num_iterations}")
    print(f"Outputs dir    : {args.output_dir}")
    print(f"Device         : {device}")
    print(line)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    envs = make_an_env(args, seed=args.seed,
                       normalize_reward=args.normalize_reward,
                       full_distribution=False)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = DQNAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,
        p_augment=args.p_augment,
        micro_dropout_p=args.micro_dropout_p
    ).to(device)

    target_network = DQNAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,
        p_augment=args.p_augment,
        micro_dropout_p=args.micro_dropout_p
    ).to(device)

    with torch.no_grad():
        example_input = 127 * np.ones((1,) + envs.single_observation_space.shape).astype(
            envs.single_observation_space.dtype)
        example_input = torch.tensor(example_input).to(device)
        q_network(example_input)
        target_network(example_input)

    # statistics, total_params, m_macs, param_bytes = network_summary(q_network, example_input, device)

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

    target_network.load_state_dict(q_network.state_dict())

    envs, q_network, global_step, b_obs = train_dqn_agent(args, logger, envs, (q_network, target_network), optimizer, device)
    envs.close()
    logger.close()
    agent = q_network

    save_checkpoint(agent, optimizer, args, global_step, envs, args.output_dir, args.run_name, 'checkpoint_final')
