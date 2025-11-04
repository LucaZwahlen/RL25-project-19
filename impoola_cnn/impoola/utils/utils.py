import time

import numpy as np
import torch
from torchinfo import summary


def network_summary(network, input_data, device):
    # Note: torchinfo accounts for masking when and params as _orig (subtracts the masked weights correctly)

    statistics = summary(
        network,
        input_data=input_data, device=device,
        depth=10,  # 2,  # 10,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "params_percent", "mult_adds"),
        verbose=1
    )
    total_params = statistics.total_params
    m_macs = np.round(statistics.total_mult_adds / 1e6, 2)
    param_bytes = statistics.total_param_bytes
    return statistics, total_params, m_macs, param_bytes


class StopTimer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0.0
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            print("Timer is already running!")

    def stop(self):
        if self.running:
            end_time = time.time()
            self.elapsed_time += end_time - self.start_time
            self.running = False
        else:
            print("Timer is not running!")

    def reset(self):
        self.elapsed_time = 0.0
        self.start_time = None
        self.running = False

    def get_elapsed_time(self):
        if self.running:
            current_time = time.time()
            return self.elapsed_time + (current_time - self.start_time)
        return self.elapsed_time

    def __str__(self):
        return f"Elapsed time: {self.get_elapsed_time()} seconds"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")


def get_device_name():
    device = get_device()
    return torch.cuda.get_device_name(0) if device.type == "cuda" else "XPU" if device.type == "xpu" else "CPU"


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    if t >= duration:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
