import random
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


class Augmentation:
    def __init__(self, max_shift=4, jitter_p=0.8, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15,
                 blur_p=0.2, grayscale_p=0.1, cutout_p=0.5, cutout_n=2, cutout_size=(8, 16)):
        self.max_shift = max_shift
        self.jitter_p = jitter_p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.blur_p = blur_p
        self.grayscale_p = grayscale_p
        self.cutout_p = cutout_p
        self.cutout_n = cutout_n
        self.cutout_size = cutout_size

    def random_shift(self, x):
        N, C, H, W = x.shape
        pad = self.max_shift
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        xs = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
        ys = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
        out = torch.empty_like(x)
        for i in range(N):
            out[i] = x_pad[i, :, ys[i]:ys[i] + H, xs[i]:xs[i] + W]
        return out

    def color_jitter(self, x):
        if random.random() > self.jitter_p: return x
        x = x / 255.0
        b = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * self.brightness
        c = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * self.contrast
        s = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * self.saturation
        h = (torch.rand(1, device=x.device) * 2 - 1) * self.hue
        x = torch.clamp(x, 0, 1)
        x = adjust_brightness(x, float(b))
        x = adjust_contrast(x, float(c))
        x = adjust_saturation(x, float(s))
        x = adjust_hue(x, float(h))
        return torch.clamp(x * 255.0, 0, 255)

    def gaussian_blur(self, x):
        if random.random() > self.blur_p: return x
        N, C, H, W = x.shape
        sigma = float(torch.empty(1).uniform_(0.1, 1.0))
        k = 3
        r = torch.arange(-k, k + 1, device=x.device).float()
        g = torch.exp(-0.5 * (r / sigma) ** 2)
        g = g / g.sum()
        gx = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        gy = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        x = F.conv2d(x, gx, padding=(0, k), groups=C)
        x = F.conv2d(x, gy, padding=(k, 0), groups=C)
        return x

    def grayscale(self, x):
        if random.random() > self.grayscale_p: return x
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        y = (x * w).sum(dim=1, keepdim=True)
        return y.repeat(1, 3, 1, 1)

    def cutout(self, x):
        if random.random() > self.cutout_p: return x
        N, C, H, W = x.shape
        n = self.cutout_n
        smin, smax = self.cutout_size
        for i in range(N):
            for _ in range(n):
                h = int(torch.randint(smin, smax + 1, (1,)))
                w = int(torch.randint(smin, smax + 1, (1,)))
                cy = int(torch.randint(0, H, (1,)))
                cx = int(torch.randint(0, W, (1,)))
                y0 = max(0, cy - h // 2);
                y1 = min(H, cy + h // 2)
                x0 = max(0, cx - w // 2);
                x1 = min(W, cx + w // 2)
                x[i, :, y0:y1, x0:x1] = 0
        return x

    def __call__(self, x):
        x = self.random_shift(x)
        x = self.color_jitter(x)
        x = self.gaussian_blur(x)
        x = self.grayscale(x)
        x = self.cutout(x)
        return torch.clamp(x, 0, 255)
