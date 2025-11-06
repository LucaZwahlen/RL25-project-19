import numpy as np

# TODO: Somehow load the evaluation of every possible value here, length is hardcoded and must match. 50'000

TEST_ENV_RANGE = 50_000


def try_get_optimal_test_path_length(env_name: str):
    """Return optimal path length for given environment, if known."""
    # TODO: only return this as a backup
    backup = np.ones((TEST_ENV_RANGE,), dtype=np.int32)  # avoid div by zero
    return backup


def try_get_optimal_train_path_length(env_name: str, distribution_mode: str):
    """Return optimal path length for given environment and distribution mode, if known."""
    length = 200 if distribution_mode == 'easy' else 500
    backup = np.ones((length,), dtype=np.int32)  # avoid div by zero
    return backup
