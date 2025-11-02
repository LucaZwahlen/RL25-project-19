import csv
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.optim as optim
import tyro

from impoola.train.agents import Vtrace
from impoola_cnn.impoola.eval import evaluation
from impoola_cnn.impoola.eval.normalized_score_lists import (progcen_easy_hns,
                                                             progcen_hard_hns,
                                                             progcen_hns)
from impoola_cnn.impoola.maker.make_env import make_an_env
from impoola_cnn.impoola.prune.redo import run_redo
from impoola_cnn.impoola.train.train_ppo_agent import log_metrics_to_csv
from impoola_cnn.impoola.train.train_vtrace_agent import train_vtrace_agent
from impoola_cnn.impoola.utils.utils import get_device, network_summary


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    n_episodes_rollout: int = int(2.5e3)
    deterministic_rollout: bool = False
    training_eval_ratio: float = 0.01
    normalize_reward: bool = True

    env_id: str = "fruitbot" #chaser
    distribution_mode: str = "easy"

    total_timesteps: int = int(25e6)
    learning_rate: float = 4.0e-4
    anneal_lr: bool = True

    num_envs: int = 90
    unroll_length: int = 20
    gamma: float = 0.99

    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 10.0

    vtrace_rho_bar: float = 1.0
    vtrace_c_bar: float = 2.0
    actor_batches_per_update: int = 1

    encoder_type: str = "new_version" # impala
    scale: int = 2
    pruning_type: str = "Baseline"
    weight_decay: float = 1.0e-5
    latent_space_dim: int = 512
    cnn_filters: tuple = (32, 64, 64)
    activation: str = 'silu'
    rescale_lr_by_scale: bool = True

    redo_tau: float = 0.025
    redo_interval: int = 100

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    log_interval: int = 1

    n_datapoints_csv: int = 500
    use_augmentation: bool = True


def save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, run_name, checkpoint_name):
    checkpoint_path = os.path.join(output_dir, f"{run_name}_{checkpoint_name}.pt")
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
    args.batch_size = int(args.num_envs * args.unroll_length)
    args.minibatch_size = args.batch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    output_dir = f"outputs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    sit_format_file = os.path.join(output_dir, "sit_format.csv")

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['global_step', 'metric', 'value'])

    with open(sit_format_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['losses/action_loss', 'losses/dist_entropy', 'losses/value_loss', 'test/mean_episode_reward',
                         'test/median_episode_reward',
                         'train/mean_episode_reward', 'train/median_episode_reward', 'train/nupdates',
                         'train/total_num_steps', 'train/total_time'])
    global progcen_hns
    if args.distribution_mode == "easy":
        progcen_hns.update(progcen_easy_hns)
    elif args.distribution_mode == "hard":
        progcen_hns.update(progcen_hard_hns)
    else:
        raise ValueError(f"Invalid distribution mode: {args.distribution_mode}")

    print(f"Run name: {run_name} | Batch size: {args.batch_size} | Num iterations: {args.num_iterations}")
    print(f"Outputs will be saved to: {output_dir}")
    print(
        f"Using SIT-style logging: losses/action_loss, losses/dist_entropy, losses/value_loss, train/test performance every epoch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = get_device()
    print(f"Device: {device}")

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
        width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False
    ).to(device)

    with torch.no_grad():
        example_input = 127 * np.ones((1,) + envs.single_observation_space.shape).astype(
            envs.single_observation_space.dtype)
        example_input = torch.tensor(example_input).to(device)
        agent.get_action_and_value(example_input)

    statistics, total_params, m_macs, param_bytes = network_summary(agent, example_input, device)

    initial_metrics = {
        "total_network_params": total_params,
        "total_network_m_macs": m_macs,
        "total_network_param_bytes": param_bytes,
    }
    log_metrics_to_csv(metrics_file, 0, initial_metrics)

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
        print(f"Rescaled learning rate to {optimizer.param_groups[0]['lr']}")

    save_checkpoint(agent, optimizer, args, 0, envs, output_dir, run_name, "checkpoint_000_initial")

    envs, agent, global_step, b_obs = train_vtrace_agent(args, envs, agent, optimizer, device)

    if args.normalize_reward:
        train_envs_return_norm_mean = envs.return_rms.mean
        train_envs_return_norm_var = envs.return_rms.var
        train_envs_return_norm_count = envs.return_rms.count

    save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, run_name, "checkpoint_100_final")

    envs.close()

    redo_dict = run_redo(b_obs[:args.minibatch_size], agent, optimizer, args.redo_tau, False, False)

    dormant_metrics = {
        "final/zero_fraction": redo_dict['zero_fraction'],
        "final/dormant_fraction": redo_dict['dormant_fraction'],
    }
    for i, (k, v) in enumerate(redo_dict['dormant_neurons_per_layer'].items()):
        dormant_metrics[f"final/dormant_neurons_{i}_{k}"] = v

    log_metrics_to_csv(metrics_file, global_step, dormant_metrics)

    agent = agent.to(device)
    print(f"Zero fraction: {redo_dict['zero_fraction']:.2f} | Dormant fraction: {redo_dict['dormant_fraction']:.2f}")

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("Running final detailed evaluation!")
    eval_args = deepcopy(args)

    evaluation.run_training_track(agent, eval_args, global_step)
    save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, run_name, "checkpoint_after_training_eval")

    evaluation.run_test_track(agent, eval_args, global_step)
    save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, run_name, "checkpoint_after_test_eval")

    print(f"All training and evaluation complete! Files saved to: {output_dir}")
    print(f"Training metrics logged in SIT format to: {metrics_file}")
