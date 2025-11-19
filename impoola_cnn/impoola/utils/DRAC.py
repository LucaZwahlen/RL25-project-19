import torch
import torch.nn as nn

HFLIP_MAP = {0: 6, 1: 7, 2: 8, 6: 0, 7: 1, 8: 2}


class DRACTransformChaserFruitbot(nn.Module):
    def __init__(self, flip_dim=3):
        super().__init__()
        self.flip_dim = flip_dim

    @torch.no_grad()
    def forward(self, x_uint8):
        return torch.flip(x_uint8, dims=[self.flip_dim])


class DRACTransformColor(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        # Slightly adjust brightness/contrast
        brightness = 0.95 + 0.1 * torch.rand(1, device=self.device)  # 0.8 to 1.2
        return torch.clamp(x.float() * brightness, 0, 255).byte()

# No action remapping needed!


@torch.no_grad()
def remap_actions_hflip(actions):

    a = actions.clone()
    mask = torch.zeros_like(a, dtype=torch.bool)
    result = a.clone()

    for src, dst in HFLIP_MAP.items():
        src_mask = (a == src) & ~mask
        result[src_mask] = dst
        mask |= src_mask

    return result


def remap_logprobs_for_flip(dist, actions):
    return dist.log_prob(remap_actions_hflip(actions))
