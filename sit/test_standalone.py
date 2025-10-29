import os
import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
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

import os
import sys
from typing import TypedDict

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_remove_dict_obs import VecExtractDictObs
from procgen import ProcgenEnv
from ucb_rl2_meta import utils
from ucb_rl2_meta.envs import TransposeImageProcgen, VecPyTorchProcgen


class RenderConfig(TypedDict):
    num_levels: int
    start_level: int
    seed: int
    speed: int


class RenderCanvas(TypedDict):
    plt: any
    axis: any
    img_plot: any


class GenericActor(ABC):
    @abstractmethod
    def act(self, obs, eval_masks):
        pass


class SitActor(GenericActor):
    def __init__(self, actor_critic, device):
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.actor_critic.eval()
        self.device = device
        self.recurrent_hidden_states = torch.zeros(
            1, actor_critic.recurrent_hidden_state_size, device=device)

    def act(self, obs, eval_masks):
        _, action, _, rec_hidden = self.actor_critic.act(
            obs,
            self.recurrent_hidden_states,
            eval_masks,
            deterministic=False
        )
        self.recurrent_hidden_states = rec_hidden
        return action


def render(args, actor, device, config: RenderConfig, canvas: RenderCanvas, aug_id=None):

    # Sample Levels From the Full Distribution
    venv = ProcgenEnv(num_envs=1, env_name=args.env_name,
                      num_levels=config['num_levels'], start_level=config['start_level'], rand_seed=config['seed'],
                      distribution_mode=args.distribution_mode)  # Remove render_mode
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=False)  # Remove reward normalization for rendering
    eval_envs = VecPyTorchProcgen(venv, device)

    obs = eval_envs.reset()

    eval_masks = torch.ones(1, 1, device=device)

    # Setup rendering with matplotlib if requested
    canvas['axis'].set_title(f"Playing {args.env_name} - Reward: 0")

    envs_done = 0
    curr_env_reward = 0.0

    while envs_done < 10:
        with torch.no_grad():
            obs_aug = aug_id(obs) if aug_id else obs
            action = actor.act(obs_aug, eval_masks)

        obs, reward, done, infos = eval_envs.step(action)
        curr_env_reward += reward.item()

        # Render the game using matplotlib
        # Convert observation to displayable format
        # obs is shape (1, 3, 64, 64), convert to (64, 64, 3)
        rgb_frame = obs[0].cpu().numpy().transpose(1, 2, 0)
        # Normalize to 0-1 range for display
        rgb_frame = (rgb_frame - rgb_frame.min()) / (rgb_frame.max() - rgb_frame.min())

        canvas['img_plot'].set_array(rgb_frame)
        canvas['axis'].set_title(f"Playing {args.env_name} - Reward: {curr_env_reward:.1f}")

        plt.pause(1/config['speed'])  # Slow down for viewing

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                print(f"Episode finished with reward: {info['episode']['r']:.1f}")
                envs_done += 1
                curr_env_reward = 0.0

    eval_envs.close()


def load_sit_checkpoint(checkpoint_path: str, device: torch.device, args, shape=(3, 64, 64), action_space=15) -> DrAC:
    print(f"Loading checkpoint from {checkpoint_path}")
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    actor_critic = Policy_Sit(
        shape,
        action_space,
        device,
        hidden_size=args.hidden_size,
        choice=args.choice,
        base_kwargs={'recurrent': False})

    actor_critic.to(device)

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

    print(f"Loaded SIT model from epoch {checkpoint.get('epoch', 'unknown')}")

    return agent


# def load_impoola_ppo_checkpoint(checkpoint_path: str, device: torch.device, shape=(3, 64, 64), action_space=15) -> PPOAgent:
#     print(f"Loading checkpoint from {checkpoint_path}")
#     # Load checkpoint
#     if not os.path.exists(checkpoint_path):
#         print(f"Checkpoint not found.")
#         sys.exit(1)

#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     args = checkpoint['args']
#     agent = PPOAgent(
#         encoder_type=args.encoder_type,
#         envs=None,
#         width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
#         activation=args.activation,
#         use_layer_init_normed=False

#     ).to(device)

#     # Load weights exactly like train2.py does it
#     agent.load_state_dict(checkpoint['agent_state_dict'])

#     print(f"Loaded IMPOOLA model from step {checkpoint.get('step', 'unknown')}")

#     return agent


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

    args = parser.parse_args()

    # Set device same as train2.py
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load the trained agent
    agent = load_sit_checkpoint(args.checkpoint, device, args)
    # agent = load_impoola_ppo_checkpoint(args.checkpoint, device)
    # Set up augmentation same as train2.py
    aug_id = data_augs.Identity

    render_config: RenderConfig = {
        'num_levels': 0,
        'start_level': 0,
        'seed': 42,
        'speed': 50,
    }

    # plot
    plt.ion()
    fig, ax = plt.subplots()

    canvas_config: RenderCanvas = {
        'plt': plt,
        'axis': ax,
        'img_plot': ax.imshow(np.zeros((64, 64, 3))),
    }

    sit_actor = SitActor(agent.actor_critic, device)

    render(args, sit_actor, device, render_config, canvas_config, aug_id=aug_id)
