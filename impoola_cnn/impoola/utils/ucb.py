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
