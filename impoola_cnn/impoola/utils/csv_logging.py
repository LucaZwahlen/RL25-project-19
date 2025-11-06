import csv
import json
import math
import os
import time
from collections import deque

import numpy as np
import torch

from impoola_cnn.impoola.utils.environment_knowledge import (
    try_get_optimal_test_path_length, try_get_optimal_train_path_length)
from impoola_cnn.impoola.utils.noop_indices import get_noop_indices
from impoola_cnn.impoola.utils.success_rewards import get_success_reward

EPSILON = 0.25


class EpisodeQueueCalculator:
    def __init__(self, is_train, normalize_reward, queue_len, env_id, num_envs, distribution_mode, device):
        self.normalize_reward = normalize_reward
        self.optimal_path_length = try_get_optimal_train_path_length(env_id, distribution_mode) if is_train else try_get_optimal_test_path_length(env_id)

        self.ticks = torch.zeros((num_envs,), device=device)
        self.steps = torch.zeros((num_envs,), device=device)
        self.success = torch.zeros((num_envs,), dtype=torch.bool, device=device)

        if queue_len == 0:
            self.rewards = []
            self.level_seeds = []
            self.num_ticks = []
            self.num_steps = []
            self.is_success = []
            self.spl_terms = []
        else:
            self.rewards = deque(maxlen=queue_len)
            self.level_seeds = deque(maxlen=queue_len)
            self.num_ticks = deque(maxlen=queue_len)
            self.num_steps = deque(maxlen=queue_len)
            self.is_success = deque(maxlen=queue_len)
            self.spl_terms = deque(maxlen=queue_len)

        self.success_reward = torch.tensor(get_success_reward(env_id))

        noop_indices = get_noop_indices(env_id)
        self.noop_mask = torch.zeros((15,), dtype=torch.bool, device=device)
        for idx in noop_indices:
            self.noop_mask[idx] = 1

    def update(self, action, rewards):
        # here we assume that when normalizing, the completed reward is large enough that it will go to the clamp max (10.0)
        success_reward_fixed = torch.tensor(10.0) if self.normalize_reward else self.success_reward
        self.ticks += 1
        action_op_mask = ~(self.noop_mask[action])
        self.steps += action_op_mask
        self.success = self.success | (rewards >= success_reward_fixed - EPSILON)

    def extend(self, info):
        completed_episodes = info["episode"]["r"][info["_episode"]]
        self.rewards.extend(completed_episodes)
        completed_levels_prev_seeds = np.array(info['prev_level_seed'])[info["_episode"]]
        self.level_seeds.extend(completed_levels_prev_seeds)
        completed_ticks = self.ticks[info["_episode"]].cpu().numpy()
        self.num_ticks.extend(completed_ticks)
        completed_steps = self.steps[info["_episode"]].cpu().numpy()
        self.num_steps.extend(completed_steps)
        completed_episode_succeeded = self.success[info["_episode"]].cpu().numpy()
        self.is_success.extend(completed_episode_succeeded)
        self.ticks[info["_episode"]] = 0
        self.steps[info["_episode"]] = 0
        self.success[info["_episode"]] = 0

        # spl
        L_star = self.optimal_path_length[completed_levels_prev_seeds]
        t_i = completed_episode_succeeded * (L_star / np.maximum(completed_steps, L_star))
        self.spl_terms.extend(t_i)

    def get_statistics(self):
        mean_reward = np.mean(self.rewards) if len(self.rewards) > 0 else 0.0
        median_reward = np.median(self.rewards) if len(self.rewards) > 0 else 0.0
        mean_ticks = np.mean(self.num_ticks) if len(self.num_ticks) > 0 else 0.0
        mean_steps = np.mean(self.num_steps) if len(self.num_steps) > 0 else 0.0
        mean_success = np.mean(self.is_success) if len(self.is_success) > 0 else 0.0
        mean_spl = np.mean(self.spl_terms) if len(self.spl_terms) > 0 else 0.0
        levels = list(self.level_seeds)
        count = len(self.rewards)

        return mean_reward, median_reward, mean_ticks, mean_steps, mean_success, mean_spl, levels, count


class Logger:
    def __init__(self, args):
        self.output_file_csv = os.path.join(args.output_dir, "sit_format.csv")
        self.output_file_levels = os.path.join(args.output_dir, "levels.jsonl")
        self.output_file_config = os.path.join(args.output_dir, "config.json")

        # Initialize files and directories
        os.makedirs(args.output_dir, exist_ok=True)
        self._save_config_file(args)

        # Open CSV file and keep handle open
        self.csv_file = open(self.output_file_csv, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Open levels file and keep handle open
        self.levels_file = open(self.output_file_levels, 'w')

        # Write CSV header
        self.csv_writer.writerow(
            [
                "loss/action_loss",
                "loss/dist_entropy",
                "loss/value_loss",
                "test/mean_reward",
                "test/median_reward",
                "test/count",
                "test/ticks",
                "test/steps",
                "test/success",
                "test/spl",
                "train/mean_reward",
                "train/median_reward",
                "train/count",
                "train/ticks",
                "train/steps",
                "train/success",
                "train/spl",
                "nupdates",
                "total_steps",
                "training_time"
            ]
        )
        self.csv_file.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _fmt_num(self, x):
        if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)):
            return f"{float(x):.4f}"
        return x

    def _save_config_file(self, args):
        with open(self.output_file_config, 'w') as f:
            json.dump(vars(args), f, indent=2)

    def _log_level_data(self, level_data):
        json.dump(level_data, self.levels_file)
        self.levels_file.write('\n')
        self.levels_file.flush()

    def _log_csv_data(self, row):
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def log(self, action_loss, dist_entropy, value_loss, test_mean, test_median, test_levels, test_count, test_ticks, test_steps, test_success, test_spl,
            train_mean, train_median, train_levels, train_count, train_ticks, train_steps, train_success, train_spl, nupdates, total_steps, training_time):
        row = [
            self._fmt_num(action_loss),
            self._fmt_num(dist_entropy),
            self._fmt_num(value_loss),
            self._fmt_num(test_mean),
            self._fmt_num(test_median),
            test_count,
            self._fmt_num(test_ticks),
            self._fmt_num(test_steps),
            self._fmt_num(test_success),
            self._fmt_num(test_spl),
            self._fmt_num(train_mean),
            self._fmt_num(train_median),
            train_count,
            self._fmt_num(train_ticks),
            self._fmt_num(train_steps),
            self._fmt_num(train_success),
            self._fmt_num(train_spl),
            nupdates,
            total_steps,
            self._fmt_num(training_time),
        ]

        levels = {
            "nupdates": nupdates,
            "total_steps": total_steps,
            "test_levels": [int(lvl) for lvl in test_levels],
            "train_levels": [int(lvl) for lvl in train_levels],
        }

        self._log_csv_data(row)
        self._log_level_data(levels)

    def close(self):
        if self.levels_file:
            self.levels_file.close()
        if self.csv_file:
            self.csv_file.close()
