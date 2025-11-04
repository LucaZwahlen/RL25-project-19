import csv
import json
import math
import os
import time

import numpy as np

from impoola_cnn.impoola.utils.evaluate_test_performance import evaluate_test_performance


def _evaluate(
        total_policy_loss,
        total_value_loss,
        total_entropy_loss,
        n_updates,
        iteration,
        global_step,
        training_episode_rewards,
        agent,
        args,
        device,
        output_dir,
        iteration_start_time,
        cumulative_training_time,
):
    avg_policy_loss = total_policy_loss / n_updates
    avg_value_loss = total_value_loss / n_updates
    avg_entropy_loss = total_entropy_loss / n_updates

    iteration_end_time = time.time()
    cumulative_training_time += (iteration_end_time - iteration_start_time)

    test_mean, test_median = evaluate_test_performance(agent, args, device)

    train_mean_reward = np.mean(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0
    train_median_reward = np.median(training_episode_rewards) if len(training_episode_rewards) > 0 else 0.0

    log_sit_style_csv(
        os.path.join(output_dir, "sit_format.csv"),
        avg_policy_loss,  # action_loss
        avg_entropy_loss,  # dist_entropy
        avg_value_loss,  # value_loss
        test_mean,  # test_mean
        test_median,  # test_median
        train_mean_reward,  # train_mean
        train_median_reward,  # train_median
        iteration,  # nupdates
        global_step,  # total_steps
        cumulative_training_time  # training_time (cumulative, excluding evaluations)
    )

    iteration_start_time = time.time()

    return iteration_start_time, cumulative_training_time


def _fmt_num(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)):
        return f"{float(x):.4f}"
    return x


def log_sit_style_csv(csv_file, action_loss, dist_entropy, value_loss, test_mean, test_median, train_mean, train_median,
                      nupdates, total_steps, training_time):
    row = [
        _fmt_num(action_loss),
        _fmt_num(dist_entropy),
        _fmt_num(value_loss),
        _fmt_num(test_mean),
        _fmt_num(test_median),
        _fmt_num(train_mean),
        _fmt_num(train_median),
        nupdates,
        total_steps,
        _fmt_num(training_time),
    ]
    with open(csv_file, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def init_files(args):
    os.makedirs(args.output_dir, exist_ok=True)
    save_config_file(args)

    sit_format_file = os.path.join(args.output_dir, "sit_format.csv")

    with open(sit_format_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
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


def save_config_file(args):
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
