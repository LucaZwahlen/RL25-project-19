success_rewards = {
    "chaser": 10,
    "fruitbot": 20,
}


def get_success_reward(env_name: str):
    if env_name in success_rewards:
        return success_rewards[env_name]
    else:
        raise ValueError(f"Success reward not defined for environment: {env_name}")
