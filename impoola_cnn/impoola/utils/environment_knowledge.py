from os import path

import numpy as np

TEST_ENV_RANGE = 50_000
SCRIPT_DIR = path.dirname(path.abspath(__file__))
BASE_PATH = path.abspath(path.join(SCRIPT_DIR, '../eval/all_knowing_path_lengths/all_knowing_path_lengths'))


def try_load_file(base_name: str, env_name: str, seed: int, min_lenght: int):
    """Try to load optimal path lengths from file."""
    filename = f"{base_name}_{env_name}_{seed}.txt"
    try:
        data = np.loadtxt(filename)
        if len(data) >= min_lenght:
            return data
        else:
            print(f"Warning: loaded data from {filename} is too short (length {len(data)}), expected at least {min_lenght}. Using backup.")
            return None
    except FileNotFoundError:
        print(f"Warning: file {filename} not found. Using backup.")
        return None


def try_get_optimal_test_path_length(env_name: str, seed: int):
    """Return optimal path length for given environment, if known."""
    data = try_load_file(BASE_PATH, env_name, seed=seed, min_lenght=TEST_ENV_RANGE)
    if data is not None:
        return data

    backup = np.ones((TEST_ENV_RANGE,), dtype=np.int32)  # avoid div by zero
    return backup


def try_get_optimal_train_path_length(env_name: str, seed: int, distribution_mode: str):
    """Return optimal path length for given environment and distribution mode, if known."""
    length = 200 if distribution_mode == 'easy' else 500

    data = try_load_file(BASE_PATH, env_name, seed=seed, min_lenght=length)
    if data is not None:
        return data

    backup = np.ones((length,), dtype=np.int32)  # avoid div by zero
    return backup


def try_get_optimal_all_knowing_path_length(env_name: str, distribution_mode: str):
    """Return a list of ones..."""
    backup = np.ones((TEST_ENV_RANGE,), dtype=np.int32)  # avoid div by zero
    return backup
