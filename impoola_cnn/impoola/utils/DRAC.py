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

@torch.no_grad()
def remap_actions_hflip(actions):
    a = actions.clone()
    for src, dst in HFLIP_MAP.items():
        a[a == src] = dst
    return a

def remap_logprobs_for_flip(dist, actions):
    return dist.log_prob(remap_actions_hflip(actions))
