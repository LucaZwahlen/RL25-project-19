import gym
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces.box import Box
from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb
from PIL import Image
from scipy.ndimage import gaussian_filter

from baselines.common.vec_env.vec_env import VecEnvWrapper


def fast_perceptual_downscale(
        img_arr: np.ndarray,
        factor: float = 2.0,
        # Resampling choice: BOX is very fast, LANCZOS is sharper but slower
        resample=Image.Resampling.BOX,
        # Preprocessing
        chroma_sigma_base: float = 0.15,  # small smoothing to reduce color aliasing
        luma_sigma_base: float = 0.25,  # tiny blur to stabilize gradients
        # Edge-aware boosts
        luma_boost_strength: float = 28.0,  # boost luminance near edges
        chroma_boost_strength: float = 16.0,  # boost chroma near edges for color crispness
        # Post-sharpen
        unsharp_amount: float = 0.45,
        unsharp_sigma: float = 0.6
) -> np.ndarray:
    if factor <= 0:
        raise ValueError("factor must be > 0")

    h, w = img_arr.shape[:2]
    dst_w = max(1, int(round(w / factor)))
    dst_h = max(1, int(round(h / factor)))

    # Convert to PIL for fast resize and YCbCr conversion
    src = Image.fromarray(img_arr)
    ycbcr = src.convert("YCbCr")
    y, cb, cr = [np.array(c, dtype=np.float32) for c in ycbcr.split()]

    # Lightweight pre-blur (scale-aware but cheap)
    scale = max(1.0, factor)
    luma_sigma = luma_sigma_base * scale * 0.4
    chroma_sigma = chroma_sigma_base * scale * 0.4
    if luma_sigma > 0:
        y_blur = gaussian_filter(y, sigma=luma_sigma)
    else:
        y_blur = y
    if chroma_sigma > 0:
        cb_s = gaussian_filter(cb, sigma=chroma_sigma)
        cr_s = gaussian_filter(cr, sigma=chroma_sigma)
    else:
        cb_s, cr_s = cb, cr

    # Fast gradient magnitude (cheaper than Sobel)
    # Horizontal and vertical diffs
    gx = np.diff(y_blur, axis=1, append=y_blur[:, -1:])
    gy = np.diff(y_blur, axis=0, append=y_blur[-1:, :])
    edge = np.sqrt(gx * gx + gy * gy)
    edge_norm = edge / (edge.max() + 1e-6)

    # Edge-aware boosts: luma and chroma
    y_boost = np.clip(y_blur + luma_boost_strength * edge_norm, 0, 255)
    cb_boost = np.clip(cb_s + chroma_boost_strength * edge_norm, 0, 255)
    cr_boost = np.clip(cr_s + chroma_boost_strength * edge_norm, 0, 255)

    boosted = Image.merge(
        "YCbCr",
        [
            Image.fromarray(y_boost.astype(np.uint8)),
            Image.fromarray(cb_boost.astype(np.uint8)),
            Image.fromarray(cr_boost.astype(np.uint8)),
        ],
    ).convert("RGB")

    # Fast downscale
    down = boosted.resize((dst_w, dst_h), resample=resample)

    # Mild unsharp mask on luma after downscale to restore micro-contrast
    y2, cb2, cr2 = [np.array(c, dtype=np.float32) for c in down.convert("YCbCr").split()]
    y2_blur = gaussian_filter(y2, sigma=unsharp_sigma)
    y2_sharp = np.clip(y2 + unsharp_amount * (y2 - y2_blur), 0, 255).astype(np.uint8)

    out = Image.merge(
        "YCbCr",
        [
            Image.fromarray(y2_sharp),
            Image.fromarray(cb2.astype(np.uint8)),
            Image.fromarray(cr2.astype(np.uint8)),
        ],
    ).convert("RGB")

    return np.array(out)


def fast_perceptual_downscale_gpu(
    img_tensor: torch.Tensor,
    factor: float = 2.0,
    # Preprocessing
    chroma_sigma_base: float = 0.15,
    luma_sigma_base: float = 0.25,
    # Edge-aware boosts
    luma_boost_strength: float = 28.0,
    chroma_boost_strength: float = 16.0,
    # Post-sharpen
    unsharp_amount: float = 0.45,
    unsharp_sigma: float = 0.6
) -> torch.Tensor:
    """
    GPU-accelerated perceptual downscaling matching the CPU implementation
    img_tensor: (B, C, H, W) tensor on GPU, values in [0, 255]
    """
    if factor <= 0:
        raise ValueError("factor must be > 0")

    B, C, H, W = img_tensor.shape
    dst_w = max(1, int(round(W / factor)))
    dst_h = max(1, int(round(H / factor)))

    # Convert RGB to YCbCr - img_tensor should be in [0, 1] range for kornia
    img_normalized = img_tensor / 255.0 if img_tensor.max() > 1.0 else img_tensor
    ycbcr = rgb_to_ycbcr(img_normalized)
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]

    # Convert back to [0, 255] range for processing
    y = y * 255.0
    cb = cb * 255.0
    cr = cr * 255.0

    # Lightweight pre-blur (scale-aware)
    scale = max(1.0, factor)
    luma_sigma = luma_sigma_base * scale * 0.4
    chroma_sigma = chroma_sigma_base * scale * 0.4

    if luma_sigma > 0:
        kernel_size = int(2 * round(2 * luma_sigma) + 1)
        y_blur = kornia.filters.gaussian_blur2d(y, (kernel_size, kernel_size), (luma_sigma, luma_sigma))
    else:
        y_blur = y

    if chroma_sigma > 0:
        kernel_size = int(2 * round(2 * chroma_sigma) + 1)
        cb_s = kornia.filters.gaussian_blur2d(cb, (kernel_size, kernel_size), (chroma_sigma, chroma_sigma))
        cr_s = kornia.filters.gaussian_blur2d(cr, (kernel_size, kernel_size), (chroma_sigma, chroma_sigma))
    else:
        cb_s, cr_s = cb, cr

    # Fast gradient magnitude (horizontal and vertical diffs)
    # Pad to match np.diff behavior with append
    y_blur_padded_x = F.pad(y_blur, (0, 1, 0, 0), mode='replicate')
    y_blur_padded_y = F.pad(y_blur, (0, 0, 0, 1), mode='replicate')

    gx = y_blur_padded_x[:, :, :, 1:] - y_blur_padded_x[:, :, :, :-1]
    gy = y_blur_padded_y[:, :, 1:, :] - y_blur_padded_y[:, :, :-1, :]

    edge = torch.sqrt(gx * gx + gy * gy)
    edge_norm = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    # Edge-aware boosts: luma and chroma
    y_boost = torch.clamp(y_blur + luma_boost_strength * edge_norm, 0, 255)
    cb_boost = torch.clamp(cb_s + chroma_boost_strength * edge_norm, 0, 255)
    cr_boost = torch.clamp(cr_s + chroma_boost_strength * edge_norm, 0, 255)

    # Merge back to YCbCr then RGB
    ycbcr_boosted = torch.cat([y_boost, cb_boost, cr_boost], dim=1) / 255.0
    rgb_boosted = ycbcr_to_rgb(ycbcr_boosted) * 255.0

    # Fast downscale using bilinear interpolation
    down = F.interpolate(rgb_boosted, size=(dst_h, dst_w), mode='bilinear', align_corners=False)

    # Mild unsharp mask on luma after downscale
    ycbcr_down = rgb_to_ycbcr(down / 255.0)
    y2, cb2, cr2 = ycbcr_down[:, 0:1] * 255.0, ycbcr_down[:, 1:2] * 255.0, ycbcr_down[:, 2:3] * 255.0

    kernel_size = int(2 * round(2 * unsharp_sigma) + 1)
    y2_blur = kornia.filters.gaussian_blur2d(y2, (kernel_size, kernel_size), (unsharp_sigma, unsharp_sigma))
    y2_sharp = torch.clamp(y2 + unsharp_amount * (y2 - y2_blur), 0, 255)

    # Merge final result
    ycbcr_final = torch.cat([y2_sharp, cb2, cr2], dim=1) / 255.0
    rgb_final = ycbcr_to_rgb(ycbcr_final) * 255.0

    return rgb_final


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImageProcgen(TransposeObs):
    def __init__(self, env=None, op=[0, 3, 2, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImageProcgen, self).__init__(env)
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[2], obs_shape[1], obs_shape[0]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        if ob.shape[0] == 1:
            ob = ob[0]
        return ob.transpose(self.op[0], self.op[1], self.op[2], self.op[3])


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype)

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecPyTorchProcgenSmall(VecPyTorchProcgen):
    def __init__(self, venv, device):
        super().__init__(venv, device)  # correct super
        self.device = device
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32
        )

    def _downscale_batch(self, obs_np):
        b, h, w, c = obs_np.shape
        if h == 32 and w == 32:
            return obs_np
        dst_h, dst_w = 32, 32
        out = np.empty((b, dst_h, dst_w, c), dtype=np.uint8)
        for i in range(b):
            img = Image.fromarray(obs_np[i], mode="RGB")
            down = img.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
            out[i] = np.array(down, dtype=np.uint8)
        return out

    def _downscale_batch_gpu(self, obs_tensor):
        """GPU-based perceptual downscaling"""
        B, C, H, W = obs_tensor.shape
        if H == 32 and W == 32:
            return obs_tensor
        # Calculate factor
        factor_h = H / 32
        factor_w = W / 32
        factor = (factor_h + factor_w) * 0.5

        # Apply GPU downscaling - input should be in [0, 255] range
        obs_255 = obs_tensor * 255.0
        downscaled = fast_perceptual_downscale_gpu(obs_255, factor=factor)

        # Convert back to [0, 1] range
        return downscaled / 255.0

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        obs = self._downscale_batch_gpu(obs)
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        obs = self._downscale_batch_gpu(obs)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
