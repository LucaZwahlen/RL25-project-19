import numpy as np
from collections import deque

class UCB:
    def __init__(self, hp_list, ucb_exploration_coef=0.5, ucb_window_length=10):
        self.hp_list = list(hp_list)
        self.num_hp_types = len(self.hp_list)
        self.ucb_exploration_coef = float(ucb_exploration_coef)
        self.return_action = [deque(maxlen=int(ucb_window_length)) for _ in range(self.num_hp_types)]
        self.num_action = np.ones(self.num_hp_types, dtype=float)
        self.qval_action = np.zeros(self.num_hp_types, dtype=float)
        self.expl_action = np.zeros(self.num_hp_types, dtype=float)
        self.ucb_action = np.zeros(self.num_hp_types, dtype=float)
        self.total_num = 1
        self.current_hp_id = 0

    def select_ucb_hp(self):
        for i in range(self.num_hp_types):
            self.expl_action[i] = self.ucb_exploration_coef * np.sqrt(np.log(self.total_num) / self.num_action[i])
            self.ucb_action[i] = self.qval_action[i] + self.expl_action[i]
        ucb_hp_id = int(np.argmax(self.ucb_action))
        self.current_hp_id = ucb_hp_id
        return ucb_hp_id, self.hp_list[ucb_hp_id]

    def update_ucb_values(self, returns_tensor):
        self.total_num += 1
        self.num_action[self.current_hp_id] += 1.0
        self.return_action[self.current_hp_id].append(float(returns_tensor.mean().item()))
        self.qval_action[self.current_hp_id] = float(np.mean(self.return_action[self.current_hp_id]))
