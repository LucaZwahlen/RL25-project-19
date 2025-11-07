import numpy as np
from collections import deque

class UCB:
    def __init__(self,
                 hp_list,
                 ucb_exploration_coef=0.5,
                 ucb_window_length=10):
        self.hp_list = hp_list
        self.num_hp_types = len(hp_list)
        self.ucb_exploration_coef = ucb_exploration_coef
        self.total_num = 1
        self.num_action = [1.] * self.num_hp_types
        self.qval_action = [0.] * self.num_hp_types
        self.expl_action = [0.] * self.num_hp_types
        self.ucb_action = [0.] * self.num_hp_types
        self.return_action = [deque(maxlen=ucb_window_length) for _ in range(self.num_hp_types)]
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
        self.num_action[self.current_hp_id] += 1
        self.return_action[self.current_hp_id].append(returns_tensor.mean().item())
        self.qval_action[self.current_hp_id] = float(np.mean(self.return_action[self.current_hp_id]))


import numpy as np
from collections import deque

class GaussianThompsonSampling:
    def __init__(self,
                 param_values,
                 init_mean=0.,
                 init_std=1.,
                 window_size=10):
        self.param_values = list(param_values)
        self.means = np.full(len(param_values), init_mean, dtype=float)
        self.stds = np.full(len(param_values), init_std, dtype=float)
        self.counts = np.zeros(len(param_values), dtype=int)
        self.avg_returns = deque(maxlen=window_size)
        self.current_param = None
        self.action_idx = 0

    def select_param(self):
        epsilon = 1e-6
        samples = [np.random.normal(m, max(s, epsilon)) for m, s in zip(self.means, self.stds)]
        self.action_idx = int(np.argmax(samples))
        self.current_param = self.param_values[self.action_idx]
        return self.action_idx, self.current_param

    def update_distribution(self, returns_tensor):
        self.avg_returns.append(returns_tensor.mean().item())
        reward = float(np.mean(self.avg_returns))
        index = self.param_values.index(self.current_param)
        self.counts[index] += 1
        prev_mean = float(self.means[index])
        self.means[index] = prev_mean + (reward - self.means[index]) / self.counts[index]
        if self.counts[index] > 1:
            self.stds[index] = np.sqrt(((self.stds[index] ** 2) * (self.counts[index] - 1) +
                                        (reward - prev_mean) * (reward - self.means[index])) /
                                       self.counts[index])
