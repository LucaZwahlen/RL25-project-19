import os
import sys
import time

import numpy as np
import torch

from baselines.common.vec_env.vec_monitor import VecMonitor
from ucb_rl2_meta.algo.drac import DrAC

# Fix for numpy deprecations
if not hasattr(np, 'bool'):
    np.bool = bool
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str

from procgen import ProcgenEnv

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_remove_dict_obs import VecExtractDictObs
from ucb_rl2_meta import utils
from ucb_rl2_meta.envs import (TransposeImageProcgen, VecPyTorchProcgen,
                               VecPyTorchProcgenSmall)


def evaluate(args, actor_critic, device, num_processes=1, aug_id=None, render=False):
    actor_critic.eval()

    # Sample Levels From the Full Distribution
    venv = ProcgenEnv(num_envs=num_processes, env_name=args.env_name,
                      num_levels=200, start_level=0,
                      distribution_mode=args.distribution_mode)  # Remove render_mode
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    # eval_envs = VecPyTorchProcgen(venv, device)
    eval_envs = VecPyTorchProcgenSmall(venv, device)

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.ones(num_processes, 1, device=device)

    episode_count = 0

    # Setup rendering with matplotlib if requested
    if render and num_processes == 1:
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_title(f"Playing {args.env_name}")
            img_plot = None
            print("Game window will appear - close it to stop rendering")
        except ImportError:
            print("matplotlib not available, running without rendering")
            render = False

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            if aug_id:
                obs_aug = aug_id(obs)
            else:
                obs_aug = obs
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs_aug,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=False)

        obs, _, done, infos = eval_envs.step(action)

        # Render the game using matplotlib
        if render and num_processes == 1:
            try:
                # Convert observation to displayable format
                # obs is shape (1, 3, 64, 64), convert to (64, 64, 3)
                rgb_frame = obs[0].cpu().numpy().transpose(1, 2, 0)
                # Normalize to 0-1 range for display
                rgb_frame = (rgb_frame - rgb_frame.min()) / (rgb_frame.max() - rgb_frame.min())

                if img_plot is None:
                    img_plot = ax.imshow(rgb_frame)
                    ax.axis('off')
                else:
                    img_plot.set_array(rgb_frame)

                plt.pause(0.05)  # Slow down for viewing

                # Check if window was closed
                if not plt.get_fignums():
                    render = False

            except Exception as e:
                print(f"Rendering error: {e}")
                render = False

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                episode_count += 1
                print(f"Episode {episode_count} finished with reward: {info['episode']['r']:.1f}")

    eval_envs.close()

    if render:
        plt.close('all')

    print("Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n"
          .format(len(eval_episode_rewards),
                  np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))

    return eval_episode_rewards


# Standalone testing when run as main
if __name__ == "__main__":
    import os
    import sys

    # Add the train2.py arguments by importing them
    sys.path.append(os.path.dirname(__file__))
    import data_augs
    from train2 import \
        parser  # This gets the parser with all train2.py arguments
    from ucb_rl2_meta.model import Policy_Sit

    # Add test-specific arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--render', action='store_true', help='Render the environment')

    args = parser.parse_args()

    # Set device same as train2.py
    device = torch.device("cuda:" + str(args.device_id))

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model exactly like train2.py
    obs_shape = (3, 64, 64)  # Procgen obs shape
    action_space_n = 15  # Procgen action space

    if args.use_sit:
        actor_critic = Policy_Sit(
            obs_shape,
            action_space_n,
            device,
            hidden_size=args.hidden_size,
            choice=args.choice,
            base_kwargs={'recurrent': False})
    else:
        from ucb_rl2_meta.model import Policy
        actor_critic = Policy(
            obs_shape,
            action_space_n,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size})

    actor_critic.to(device)

    # Create agent exactly like train2.py
    aug_id = data_augs.Identity
    aug_list = []
    agent = DrAC(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        aug_id=aug_id,
        aug_func=aug_list,
        aug_type=args.aug_choice,
        aug_coef=args.aug_coef,
        env_name=args.env_name)

    # Load weights exactly like train2.py does it
    agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Set up augmentation same as train2.py
    aug_id = data_augs.Identity

    # Run evaluation using the actor_critic directly like train2.py does
    if args.render:
        print("Rendering enabled - game window should appear")
        num_processes = 1  # Use single process for rendering
    else:
        num_processes = 4

    print("Starting evaluation...")
    # Use actor_critic directly, not agent, like train2.py does in the evaluate call
    rewards = evaluate(args, agent.actor_critic, device, num_processes=num_processes,
                       aug_id=aug_id, render=args.render)

    print(f"Evaluation complete. Mean reward: {np.mean(rewards):.2f}")
