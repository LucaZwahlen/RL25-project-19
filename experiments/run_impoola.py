import argparse
import os

import numpy as np
import torch

from impoola_cnn.impoola.maker.make_env import make_procgen_env
from impoola_cnn.impoola.train.agents import PPOAgent, Vtrace
from impoola_cnn.impoola.utils.save_load import load_checkpoint
from impoola_cnn.impoola.utils.utils import get_device


def print_run_config(args, device, model_found, extra=None):
    line = "=" * 80
    sub = "-" * 80
    title = "TRAINING RUN CONFIGURATION"
    print(line)
    print(f"{title:^80}")
    print(sub)
    print(f"Env id                : {args.env_id}")
    print(f"Distribution mode     : {args.distribution_mode}")
    print(f"Num envs              : {args.num_envs}")
    print(f"Episodes              : {args.episodes}")
    print(f"Encoder type          : {args.encoder_type}")
    print(f"Scale                 : {args.scale}")
    print(f"Latent space dim      : {args.latent_space_dim}")
    print(f"CNN filters           : {list(args.cnn_filters)}")
    print(f"Activation            : {args.activation}")
    print(f"Deterministic         : {args.deterministic}")
    print(f"Checkpoint            : {args.checkpoint or '(none)'}")
    print(f"Agent                 : {args.agent}")
    print(f"Device                : {device}")
    if extra:
        for k, v in extra.items():
            print(f"{k:<22}: {v}")
    print(sub)
    if model_found:
        print("Model file            : FOUND")
    else:
        warn = "!!! CHECKPOINT NOT FOUND - RUNNING WITHOUT A LOADED MODEL !!!"
        pad = (80 - len(warn)) // 2
        print("=" * 80)
        print(" " * pad + warn)
        print("=" * 80)
    print(line)


def build_agent(args, envs, device):
    common = dict(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale,
        out_features=args.latent_space_dim,
        cnn_filters=tuple(args.cnn_filters),
        activation=args.activation,
        use_layer_init_normed=False,
    )
    if args.agent.lower() == "ppo":
        agent = PPOAgent(**common).to(device)
    elif args.agent.lower() == "vtrace":
        agent = Vtrace(**common).to(device)
    else:
        raise ValueError("agent must be PPO or Vtrace")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="fruitbot")
    parser.add_argument(
        "--distribution_mode", type=str, default="easy", choices=["easy", "hard"]
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--encoder_type", type=str, default="impala")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--latent_space_dim", type=int, default=256)
    parser.add_argument("--cnn_filters", type=int, nargs="+", default=[16, 32, 32])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--agent", type=str, default="Vtrace")
    args = parser.parse_args()

    class TempArgs:
        pass

    targs = TempArgs()
    targs.env_id = args.env_id
    targs.num_envs = args.num_envs
    targs.gamma = 0.99

    envs = make_procgen_env(
        targs,
        full_distribution=False,
        normalize_reward=False,
        rand_seed=1,
        render=True,
        distribution_mode=args.distribution_mode,
    )

    device = get_device()
    agent = build_agent(args, envs, device)

    model_found = bool(args.checkpoint) and os.path.isfile(args.checkpoint)
    if model_found:
        dummy_optim = torch.optim.Adam(agent.parameters(), lr=1e-4)
        load_checkpoint(agent, dummy_optim, args.checkpoint, device, envs=None)

    print_run_config(
        args=args,
        device=device,
        model_found=model_found,
        extra={
            "Action space": getattr(
                getattr(envs, "single_action_space", None), "n", "?"
            )
        },
    )

    obs, _ = envs.reset()
    obs = torch.as_tensor(obs, device=device)
    done = np.zeros(args.num_envs, dtype=bool)
    ep_count = 0

    with torch.no_grad():
        while ep_count < args.episodes:
            if model_found:
                if args.deterministic:
                    pi, _ = agent.get_pi_and_value(obs)
                    action = pi.mode
                else:
                    action, _, _, _, _ = agent.get_action_and_value(obs)
            else:
                action = torch.randint(
                    0, envs.single_action_space.n, (args.num_envs,), device=device
                )

            next_obs, reward, terminated, truncated, info = envs.step(
                action.detach().cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            obs = torch.as_tensor(next_obs, device=device)
            if done.any():
                ep_count += int(done.sum())

    envs.close()


if __name__ == "__main__":
    main()
