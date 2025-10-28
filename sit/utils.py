import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")


def get_device_name():
    device = get_device()
    return torch.cuda.get_device_name(0) if device.type == "cuda" else "XPU" if device.type == "xpu" else "CPU"
