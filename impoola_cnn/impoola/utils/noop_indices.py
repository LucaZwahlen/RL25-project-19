# the indices of actions that correspond to no-ops in various Procgen games
# note the shared action space:
# [('LEFT', 'DOWN'), ('LEFT',), ('LEFT', 'UP'), ('DOWN',), (), ('UP',), ('RIGHT', 'DOWN'), ('RIGHT',), ('RIGHT', 'UP'), ('D',), ('A',), ('W',), ('S',), ('Q',), ('E',)]
# now go check the implementation of the games for what actions they use, count everything else as a no-op

# hint:
# vx is nonzero for: [0, 1, 2, 6, 7, 8]
# vy is nonzero for: [0, 2, 3, 5, 6, 8]
# move_act is a number from 0-8 corresponding to the first 9 actions above, where 4 is no-op
# special actions are 9-14, as a nuber from 1-6, setting the move_action to 4 always.

# fruitbot:
# only action_vx matters for direction, vy is overridden to 0.2. only special action 1 matters (fire)
# https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/games/fruitbot.cpp

# chaser:
# only vx vy matters, no specials
# https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/games/chaser.cpp

noop_indices = {
    "fruitbot": [3, 4, 5, 10, 11, 12, 13, 14],
    "chaser": [4, 9, 10, 11, 12, 13, 14],
}


def get_noop_indices(env_name: str):
    if env_name in noop_indices:
        return noop_indices[env_name]
    else:
        raise ValueError(f"No-op indices not defined for environment: {env_name}")
