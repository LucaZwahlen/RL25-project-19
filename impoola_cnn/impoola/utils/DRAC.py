import torch
import torch.nn as nn

UP = [5, 11]
DOWN = [3, 12]
LEFT = [1, 10]
RIGHT = [7, 9]


def _build_swap_map(group_a, group_b):
    assert len(group_a) == len(group_b)
    return {a: b for a, b in zip(group_a, group_b)} | {b: a for a, b in zip(group_a, group_b)}


HFLIP_MAP = _build_swap_map(LEFT, RIGHT)
VFLIP_MAP = _build_swap_map(UP, DOWN)


class DRACTransformChaserFruitbot(nn.Module):
    def __init__(self, hflip=False, vflip=False):
        super().__init__()
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)

    @torch.no_grad()
    def _maybe_flip(self, x):
        if self.hflip:
            x = torch.flip(x, dims=[3])
        if self.vflip:
            x = torch.flip(x, dims=[2])
        return x

    @torch.no_grad()
    def forward(self, x_uint8):
        x = x_uint8.float() / 255.0
        x = self._maybe_flip(x)
        return (x.clamp(0.0, 1.0) * 255.0).round().to(dtype=x_uint8.dtype)


@torch.no_grad()
def _remap_with_map(actions, swap_map):
    a = actions.clone()
    if len(swap_map) == 0:
        return a
    for src, dst in swap_map.items():
        mask = (a == src)
        if mask.any():
            a[mask] = dst
    return a


def remap_actions_hflip(actions):
    return _remap_with_map(actions, HFLIP_MAP)


def remap_actions_vflip(actions):
    return _remap_with_map(actions, VFLIP_MAP)


def remap_logprobs_for_flip(dist, actions, hflip=False, vflip=False):
    a = actions
    if hflip:
        a = remap_actions_hflip(a)
    if vflip:
        a = remap_actions_vflip(a)
    return dist.log_prob(a)
