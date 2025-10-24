import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class DrAC():
    """
    Data-regularized Actor-Critic (DrAC) object
    """

    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 aug_id=None,
                 aug_func=None,
                 aug_type=0,
                 aug_coef=0.1,
                 env_name=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.scheduler = StepLR(self.optimizer, step_size=20000, gamma=0.9)  # 0.9

        self.aug_id = aug_id
        self.aug_func = aug_func
        self.aug_coef = aug_coef
        self.aug_type = aug_type

        self.env_name = env_name

        self.scaler = torch.cuda.amp.GradScaler()  # FP16

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            # start_time = time.perf_counter()
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                # time_1 = time.perf_counter()
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                with torch.cuda.amp.autocast():  # FP16
                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)

                    # time_2 = time.perf_counter()
                    # print(f"Time for forward pass 1: {time_2 - time_1:.4f} seconds")

                    ratio = torch.exp(action_log_probs -
                                      old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                    # time_3 = time.perf_counter()
                    # print(f"Time for loss calculation: {time_3 - time_2:.4f} seconds")

                    current_aug_func = self.aug_func[self.aug_type]  # np.random.randint(0, len(self.aug_func)-2) ]
                    obs_batch_aug = current_aug_func.do_augmentation(obs_batch)
                    # ---ad random conv by default ----
                    # obs_batch_aug =  self.aug_func[-1].do_augmentation(obs_batch_aug)
                    # obs_batch_aug =  self.aug_func[-2].do_augmentation(obs_batch_aug)
                    # -----------
                    obs_batch_id = self.aug_id(obs_batch)

                    # time_4 = time.perf_counter()
                    # print(f"Time for augmentation: {time_4 - time_3:.4f} seconds")

                    _, new_actions_batch, _, _ = self.actor_critic.act(
                        obs_batch_id, recurrent_hidden_states_batch, masks_batch)
                    values_aug, action_log_probs_aug, dist_entropy_aug, _ = \
                        self.actor_critic.evaluate_actions(obs_batch_aug,
                                                           recurrent_hidden_states_batch, masks_batch, new_actions_batch)

                # time_5 = time.perf_counter()
                # print(f"Time for double forward pass for augmentation: {time_5 - time_4:.4f} seconds")

                # Compute Augmented Loss
                action_loss_aug = - action_log_probs_aug.mean()
                value_loss_aug = .5 * (torch.detach(values) - values_aug).pow(2).mean()

                # time_6 = time.perf_counter()
                # print(f"Time for augmented loss calculation: {time_6 - time_5:.4f} seconds")

                aug_loss = value_loss_aug + action_loss_aug
                total_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + aug_loss * self.aug_coef

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()  # scale for FP16 safety
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                # time_7 = time.perf_counter()
                # print(f"Time for backward pass and optimizer step: {time_7 - time_6:.4f} seconds")

                if self.aug_func:
                    for i in range(len(self.aug_func)):
                        self.aug_func[i].change_randomization_params_all()

            # end_time = time.perf_counter()
            # print(f"Total update time for 1 PPO epoch: {end_time - start_time:.4f} seconds")

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
