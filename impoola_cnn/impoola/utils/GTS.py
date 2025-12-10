from collections import deque

import numpy as np


class GaussianThompsonSampling:
    def __init__(self, param_values, init_mean=0.0, init_std=1.0, window_size=10):
        self.param_values = list(param_values)
        self.means = np.full(len(self.param_values), init_mean, dtype=float)
        self.stds = np.full(len(self.param_values), init_std, dtype=float)
        self.counts = np.zeros(len(self.param_values), dtype=int)
        self.avg_returns = deque(maxlen=window_size)
        self.current_idx = 0

    def select_param(self):
        eps = 1e-6
        samples = [
            np.random.normal(m, max(s, eps)) for m, s in zip(self.means, self.stds)
        ]
        self.current_idx = int(np.argmax(samples))
        return self.current_idx, self.param_values[self.current_idx]

    def update_distribution(self, returns_tensor):
        r = float(np.mean(returns_tensor.detach().cpu().numpy()))
        self.avg_returns.append(r)
        reward = float(np.mean(self.avg_returns))
        i = self.current_idx
        self.counts[i] += 1
        prev_mean = self.means[i]
        self.means[i] = prev_mean + (reward - self.means[i]) / self.counts[i]
        if self.counts[i] > 1:
            self.stds[i] = np.sqrt(
                (
                    (self.stds[i] ** 2) * (self.counts[i] - 1)
                    + (reward - prev_mean) * (reward - self.means[i])
                )
                / self.counts[i]
            )
