import os
import time
from collections import deque

import numpy as np
import torch
from procgen import ProcgenEnv

import sit.data_augs
from sit.baselines import logger
from sit.baselines.common.vec_env.vec_monitor import VecMonitor
from sit.baselines.common.vec_env.vec_normalize import VecNormalize
from sit.baselines.common.vec_env.vec_remove_dict_obs import VecExtractDictObs
from sit.test import evaluate
from sit.ucb_rl2_meta import algo, utils
from sit.ucb_rl2_meta.algo.drac import DrAC
from sit.ucb_rl2_meta.arguments import parser
from sit.ucb_rl2_meta.envs import VecPyTorchProcgen
from sit.ucb_rl2_meta.model import AugCNN, Policy, Policy_Sit
from sit.ucb_rl2_meta.storage import RolloutStorage

parser.add_argument(
    '--device_id',
    type=int,
    default=1,
    help='device')

parser.add_argument(
    '--choice',
    type=int,
    default=0,
    help='whihc sit model to use 0 SiT, 1 for SiTs'
)

parser.add_argument(
    '--aug_choice',
    type=int,
    default=0,
    help='which aug-type selected')

parser.add_argument(
    '--use_sit',
    default=True,
    help='use Sit model')

parser.add_argument(
    '--use_ppo',
    default=False,
    help='use PPo algo')

# 'rotate': data_augs.Rotate_degree,
# 'crop2': data_augs.Crop,
aug_to_func = {
    'crop': sit.data_augs.Crop,
    'random-conv': sit.data_augs.RandomConv,
    # 'grayscale': sit.data_augs.Grayscale,

    # 'rotate': sit.data_augs.Rotate,
    # 'cutout': sit.data_augs.Cutout,
    # 'cutout-color': sit.data_augs.CutoutColor,
    'color-jitter': sit.data_augs.ColorJitter,
    'flip': sit.data_augs.Flip,
}


def train(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    # torch.set_num_threads(1)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:" + str(args.device_id))
    print("-------  device: ", args.device_id, "------")
    #  = torch.device("cuda:" + str(args.device_id))

    log_file = '-{}-{}-reproduce-s{}'.format(args.run_name, args.env_name, args.seed)

    venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name,
                      num_levels=args.num_levels, start_level=args.start_level,
                      distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)

    obs_shape = envs.observation_space.shape
    if args.use_sit:
        actor_critic = Policy_Sit(
            obs_shape,
            envs.action_space.n,
            device,
            hidden_size=args.hidden_size,
            choice=args.choice,
            base_kwargs={'recurrent': False})  # , 'hidden_size': args.hidden_size})
        if args.choice == 0:
            args.log_dir = "logs"
        elif args.choice == 1:
            args.log_dir = "logs"

    else:
        actor_critic = Policy(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size})

    actor_critic.to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              aug_type=args.aug_type, split_ratio=args.split_ratio)

    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    if args.use_ucb:
        aug_id = sit.data_augs.Identity
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]

        agent = UCBDrAC(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            aug_list=aug_list,
            aug_id=aug_id,
            aug_coef=args.aug_coef,
            num_aug_types=len(list(aug_to_func.keys())),
            ucb_exploration_coef=args.ucb_exploration_coef,
            ucb_window_length=args.ucb_window_length)

    elif args.use_meta_learning:
        aug_id = sit.data_augs.Identity
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]

        aug_model = AugCNN()
        aug_model.to(device)

        agent = algo.MetaDrAC(
            actor_critic,
            aug_model,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            meta_grad_clip=args.meta_grad_clip,
            meta_num_train_steps=args.meta_num_train_steps,
            meta_num_test_steps=args.meta_num_test_steps,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            aug_id=aug_id,
            aug_coef=args.aug_coef)

    elif args.use_rl2:
        aug_id = sit.data_augs.Identity
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]

        rl2_obs_shape = [envs.action_space.n + 1]
        rl2_learner = Policy(
            rl2_obs_shape,
            len(list(aug_to_func.keys())),
            base_kwargs={'recurrent': True, 'hidden_size': args.rl2_hidden_size})
        rl2_learner.to(device)

        agent = algo.RL2DrAC(
            actor_critic,
            rl2_learner,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            args.rl2_entropy_coef,
            lr=args.lr,
            eps=args.eps,
            rl2_lr=args.rl2_lr,
            rl2_eps=args.rl2_eps,
            max_grad_norm=args.max_grad_norm,
            aug_list=aug_list,
            aug_id=aug_id,
            aug_coef=args.aug_coef,
            num_aug_types=len(list(aug_to_func.keys())),
            recurrent_hidden_size=args.rl2_hidden_size,
            num_actions=envs.action_space.n,
            device=device)
    elif args.use_ppo:
        aug_id = sit.data_augs.Identity
        # aug_func = aug_to_func[args.aug_type](batch_size=batch_size)
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]
        agent = algo.PPO(
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
            aug_coef=args.aug_coef,
            env_name=args.env_name)
    else:
        aug_id = sit.data_augs.Identity
        # aug_func = aug_to_func[args.aug_type](batch_size=batch_size)
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]
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

    checkpoint_path = os.path.join(args.save_dir, "agent" + log_file + ".pt")
    if os.path.exists(checkpoint_path) and args.preempt:
        checkpoint = torch.load(checkpoint_path)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch'] + 1
        logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], log_suffix=log_file + "-e%s" % init_epoch)
    else:
        init_epoch = 0
        logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], log_suffix=log_file)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    # Initialize cumulative training timer (excluding evaluation time) - same as PPO
    cumulative_training_time = 0.0
    iteration_start_time = time.time()

    for j in range(init_epoch, num_updates):
        # get current milliseconds
        start_time_ms = time.perf_counter() * 1000
        actor_critic.train()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                obs_id = aug_id(rollouts.obs[step])
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    obs_id, rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            obs_id = aug_id(rollouts.obs[-1])
            next_value = actor_critic.get_value(
                obs_id, rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        if args.use_ucb and j > 0:
            agent.update_ucb_values(rollouts)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Update cumulative training time (excluding evaluation time)
        iteration_end_time = time.time()
        cumulative_training_time += (iteration_end_time - iteration_start_time)

        # save for every interval-th episode or for the last epoch
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("\nUpdate {}, step {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}"
                  .format(j, total_num_steps,
                          len(episode_rewards), np.mean(episode_rewards),
                          np.median(episode_rewards), dist_entropy, value_loss,
                          action_loss))

            logger.logkv("train/nupdates", j)
            logger.logkv("train/total_num_steps", total_num_steps)

            logger.logkv("losses/dist_entropy", dist_entropy)
            logger.logkv("losses/value_loss", value_loss)
            logger.logkv("losses/action_loss", action_loss)

            logger.logkv("train/mean_episode_reward", np.mean(episode_rewards))
            logger.logkv("train/median_episode_reward", np.median(episode_rewards))

            ### Eval on the Full Distribution of Levels - EXCLUDE from training time ###
            eval_start_time = time.time()
            eval_episode_rewards = evaluate(args, actor_critic, device, aug_id=aug_id)
            eval_end_time = time.time()
            # Note: We don't add eval time to cumulative_training_time

            logger.logkv("test/mean_episode_reward", np.mean(eval_episode_rewards))
            logger.logkv("test/median_episode_reward", np.median(eval_episode_rewards))

            # Add training time to the logged metrics
            logger.logkv("train/training_time", cumulative_training_time)

            logger.dumpkvs()

        # Start timing next iteration (after any potential evaluation)
        iteration_start_time = time.time()

        # Save Model
        if (j > 0 and j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            try:
                os.makedirs(args.save_dir)
            except OSError:
                pass

            torch.save({
                'epoch': j,
                'model_state_dict': agent.actor_critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, os.path.join(args.save_dir, "agent" + log_file + ".pt"))

        end_time_ms = time.perf_counter() * 1000
        print("Time for {} steps: {} ms".format(args.num_steps, end_time_ms - start_time_ms))


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
