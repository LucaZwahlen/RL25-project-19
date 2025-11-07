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
from impoola_cnn.impoola.train.agents import Vtrace
from impoola_cnn.impoola.train.train_vtrace_agent import train_vtrace_agent
from impoola_cnn.impoola.utils.csv_logging import Logger
from impoola_cnn.impoola.utils.save_load import save_checkpoint
from impoola_cnn.impoola.utils.utils import get_device


@dataclass
class Args:
    extensive_logging: bool = True  # whether to log detailed per-episode data

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    n_episodes_rollout: int = int(2.5e3)
    deterministic_rollout: bool = False
    training_eval_ratio: float = 0.01
    normalize_reward: bool = True

    env_id: str = "chaser"  # chaser
    distribution_mode: str = "easy"

    total_timesteps: int = int(25e6)
    learning_rate: float = 6.0e-4
    anneal_lr: bool = False

    num_envs: int = 80
    unroll_length: int = 20
    gamma: float = 0.99

    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True

    target_kl: Optional[float] = None
    kl_coef: float = 0.0

    vtrace_rho_bar: float = 1.0
    vtrace_c_bar: float = 2.0
    actor_batches_per_update: int = 1

    encoder_type: str = "impala_new"  # impala
    scale: int = 2
    pruning_type: str = "Baseline"
    weight_decay: float = 0.0e-5
    latent_space_dim: int = 256
    cnn_filters: tuple = (16, 32, 32)
    activation: str = 'relu'
    rescale_lr_by_scale: bool = True

    redo_tau: float = 0.025
    redo_interval: int = 100

    log_interval: int = 1

    n_datapoints_csv: int = 500

    update_epochs: int = 2
    batch_size = int(num_envs * unroll_length)
    num_iterations = total_timesteps // batch_size

    run_name: str = f"{env_id}__{exp_name}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir: str = os.path.join("outputs", run_name)

    p_augment: float = 0.0
    micro_dropout_p: float = 0.0

    drac_lambda_v: float = 1
    drac_lambda_pi: float = 0.1
    drac_vflip: bool = True
    drac_hflip: bool = True

    clip_coef: float = 0.2
    clip_vloss: bool = True


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

    if device.type == 'cuda':
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    envs = make_an_env(args, seed=args.seed,
                       normalize_reward=args.normalize_reward,
                       full_distribution=False)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Vtrace(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale,
        out_features=args.latent_space_dim,
        cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,
        p_augment=args.p_augment,
        micro_dropout_p=args.micro_dropout_p
    ).to(device)

    with torch.no_grad():
        example_input = 127 * np.ones((1,) + envs.single_observation_space.shape).astype(
            envs.single_observation_space.dtype)
        example_input = torch.tensor(example_input).to(device)
        agent.get_action_and_value(example_input)

    # statistics, total_params, m_macs, param_bytes = network_summary(agent, example_input, device)

    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        weight_decay=args.weight_decay,
        fused=True
    )

    if args.rescale_lr_by_scale:
        lr_scaling_factor = torch.tensor(args.scale / 2, device=device)
        optimizer.param_groups[0]['lr'].copy_(optimizer.param_groups[0]['lr'] / lr_scaling_factor)

    envs, agent, global_step, b_obs = train_vtrace_agent(args, logger, envs, agent, optimizer, device)

    save_checkpoint(agent, optimizer, args, global_step, envs, args.output_dir, args.run_name, "checkpoint_final")

    envs.close()
    logger.close()

    print(f"All training and evaluation complete! Files saved to: {args.output_dir}")
