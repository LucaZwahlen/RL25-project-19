import torch
import torch.nn.functional as F
import torch.nn as nn
import random

class DRACTransformChaserFruitbot(nn.Module):
    def __init__(self, crop_pad=3, p_color=0.0, brightness=0.05, contrast=0.05):
        super().__init__()
        self.crop_pad = int(crop_pad)
        self.p_color = float(p_color)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    @torch.no_grad()
    def _rand_crop(self, x):
        if self.crop_pad <= 0:
            return x
        b, c, h, w = x.shape
        x = F.pad(x, (self.crop_pad, self.crop_pad, self.crop_pad, self.crop_pad), mode='reflect')
        _, _, hp, wp = x.shape
        offs_h = torch.randint(0, hp - h + 1, (b, 1, 1), device=x.device)
        offs_w = torch.randint(0, wp - w + 1, (b, 1, 1), device=x.device)
        grid_y = torch.arange(h, device=x.device).view(1, h, 1) + offs_h
        grid_x = torch.arange(w, device=x.device).view(1, 1, w) + offs_w
        grid_y = grid_y.expand(b, h, w)
        grid_x = grid_x.expand(b, h, w)
        idx = grid_y * wp + grid_x
        x = x.view(b, c, hp * wp)
        out = torch.gather(x, 2, idx.view(b, 1, h * w).expand(b, c, h * w))
        return out.view(b, c, h, w)

    @torch.no_grad()
    def _color_jitter_bc(self, x01):
        if self.p_color <= 0 or random.random() >= self.p_color:
            return x01
        b = x01.shape[0]
        if self.brightness > 0:
            delta = (torch.rand(b,1,1,1, device=x01.device) * 2 - 1) * self.brightness
            x01 = x01 + delta
        if self.contrast > 0:
            mean = x01.mean(dim=(2,3), keepdim=True)
            factor = 1 + (torch.rand(b,1,1,1, device=x01.device) * 2 - 1) * self.contrast
            x01 = (x01 - mean) * factor + mean
        return x01.clamp(0.0, 1.0)

    @torch.no_grad()
    def forward(self, x_uint8):
        x = x_uint8.float() / 255.0
        x = self._rand_crop(x)
        x = self._color_jitter_bc(x)
        return (x.clamp(0.0,1.0) * 255.0).round().to(dtype=x_uint8.dtype)
