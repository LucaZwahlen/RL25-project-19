import csv
import json
import math
import os
import time

import numpy as np


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
            ['losses/action_loss',
             'losses/dist_entropy',
             'losses/value_loss',
             'test/mean_episode_reward',
             'test/median_episode_reward',
             'train/mean_episode_reward',
             'train/median_episode_reward',
             'train/nupdates',
             'train/total_num_steps',
             'train/total_time'
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

    def log(self, action_loss, dist_entropy, value_loss, test_mean, test_median, test_levels, test_count, test_ticks, test_steps, train_mean, train_median,
            train_levels, train_count, train_ticks, train_steps, nupdates, total_steps, training_time):
        row = [
            self._fmt_num(action_loss),
            self._fmt_num(dist_entropy),
            self._fmt_num(value_loss),
            self._fmt_num(test_mean),
            self._fmt_num(test_median),
            test_count,
            self._fmt_num(test_ticks),
            self._fmt_num(test_steps),
            self._fmt_num(train_mean),
            self._fmt_num(train_median),
            train_count,
            self._fmt_num(train_ticks),
            self._fmt_num(train_steps),
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
