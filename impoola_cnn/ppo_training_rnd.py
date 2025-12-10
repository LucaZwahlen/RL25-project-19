# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
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

from impoola_cnn.impoola.eval.normalized_score_lists import (
    progcen_easy_hns,
    progcen_hard_hns,
    progcen_hns,
)
from impoola_cnn.impoola.maker.make_env import make_an_env, make_procgen_env
from impoola_cnn.impoola.train.agents import PPOAgent
from impoola_cnn.impoola.train.train_ppo_agent_rnd import train_ppo_agent
from impoola_cnn.impoola.utils.csv_logging import Logger
from impoola_cnn.impoola.utils.environment_knowledge import TEST_ENV_RANGE
from impoola_cnn.impoola.utils.save_load import save_checkpoint
from impoola_cnn.impoola.utils.utils import get_device


@dataclass
class Args:

    extensive_logging: bool = True  # whether to log detailed per-episode data

    # all knowning
    is_all_knowing: bool = False
    move_penalty: float = 0.05

    # General Settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    n_episodes_rollout: int = int(2.5e3)
    deterministic_rollout: bool = False
    training_eval_ratio: float = 0.01
    normalize_reward: bool = True

    env_id: str = "chaser"
    distribution_mode: str = "easy"
    total_timesteps: int = int(25e6)
    learning_rate: float = 3.5e-4
    num_envs: int = 64
    num_steps: int = 256
    anneal_lr: bool = False
    gamma: float = 0.99  # Change from CleanRL, the default is 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 3
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    # Network specific arguments
    encoder_type: str = "impala"
    scale: int = 2
    pruning_type: str = "Baseline"
    weight_decay: float = 0.0e-5
    latent_space_dim: int = 256
    cnn_filters: tuple = (16, 32, 32)
    activation: str = "relu"
    rescale_lr_by_scale: bool = True

    # RND (Random Network Distillation) settings
    use_rnd: bool = True
    rnd_coef: float = 0.1  # scale of intrinsic reward added to extrinsic reward
    rnd_output_size: int = 64  # output embedding size for RND target/predictor
    rnd_lr: float = 1e-3  # learning rate for RND predictor

    # ReDo settings
    redo_tau: float = 0.025
    redo_interval: int = 100

    log_interval: int = 1
    n_datapoints_csv: int = 500

    p_augment: float = 0.0
    micro_dropout_p: float = 0.0
    drac_lambda = 0.0

    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size
    run_name = f"{env_id}__{exp_name}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir: str = os.path.join("outputs", run_name)


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

    if args.is_all_knowing:
        envs = make_procgen_env(
            args,
            full_distribution=False,
            normalize_reward=args.normalize_reward,
            rand_seed=args.seed,
            render=False,
            distribution_mode=args.distribution_mode,
            num_levels_override=TEST_ENV_RANGE,
            start_level_override=0,
        )
    else:
        envs = make_an_env(
            args,
            seed=args.seed,
            normalize_reward=args.normalize_reward,
            full_distribution=False,
        )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = PPOAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale,
        out_features=args.latent_space_dim,
        cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,
        p_augment=args.p_augment,
        micro_dropout_p=args.micro_dropout_p,
    ).to(device)

    with torch.no_grad():
        example_input = 127 * np.ones(
            (1,) + envs.single_observation_space.shape
        ).astype(envs.single_observation_space.dtype)
        example_input = torch.tensor(example_input).to(device)
        agent.get_action_and_value(example_input)

    # statistics, total_params, m_macs, param_bytes = network_summary(agent, example_input, device)

    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,  # default eps=1e-8
        weight_decay=args.weight_decay,
        fused=True,
    )

    if args.rescale_lr_by_scale:
        # LR was set for the default scale of 2, so we need to rescale it
        lr_scaling_factor = torch.tensor(args.scale / 2, device=device)
        optimizer.param_groups[0]["lr"].copy_(
            optimizer.param_groups[0]["lr"] / lr_scaling_factor
        )

    envs, agent, global_step, b_obs = train_ppo_agent(
        args, logger, envs, agent, optimizer, device
    )
    envs.close()
    logger.close()

    agent = agent.to(device)

    print("Running final detailed evaluation!")
    save_checkpoint(
        agent,
        optimizer,
        args,
        global_step,
        envs,
        args.output_dir,
        args.run_name,
        "final_checkpoint",
    )

    print(f"All training and evaluation complete! Files saved to: {args.output_dir}")
