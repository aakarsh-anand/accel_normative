from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class MaskConfig:
    mask_ratio: float = 0.6
    span_min: int = 10
    span_max: int = 200
    seed: int = 7


class SpanMasker:
    """
    Produce a boolean mask_keep [L] where True means visible to encoder.
    Masked positions will be reconstructed.
    """
    def __init__(self, cfg: MaskConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def __call__(self, L: int) -> torch.Tensor:
        keep = torch.ones(L, dtype=torch.bool)
        target_masked = int(self.cfg.mask_ratio * L)
        masked = 0
        # mask contiguous spans until target reached
        while masked < target_masked:
            span = self.rng.randint(self.cfg.span_min, self.cfg.span_max)
            start = self.rng.randint(0, max(0, L - span))
            end = min(L, start + span)
            newly = keep[start:end].sum().item()
            keep[start:end] = False
            masked += newly
            if masked >= target_masked:
                break
        return keep


def jitter(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise to accel channel"""
    # x: [B, C, L] or [C, L], only jitter accel channel (0), not mask channel (1)
    y = x.clone()
    if y.ndim == 3:
        y[:, 0] = y[:, 0] + sigma * torch.randn_like(y[:, 0])
    else:
        y[0] = y[0] + sigma * torch.randn_like(y[0])
    return y


def scale(x: torch.Tensor, lo: float = 0.8, hi: float = 1.2) -> torch.Tensor:
    """Random scaling of accel channel"""
    y = x.clone()
    s = (hi - lo) * torch.rand(1, device=x.device) + lo
    if y.ndim == 3:
        y[:, 0] = y[:, 0] * s
    else:
        y[0] = y[0] * s
    return y


def time_shift(x: torch.Tensor, max_shift: int = 50) -> torch.Tensor:
    """Randomly shift signal in time (circular shift)"""
    y = x.clone()
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if y.ndim == 3:
        y[:, 0] = torch.roll(y[:, 0], shifts=shift, dims=-1)
    else:
        y[0] = torch.roll(y[0], shifts=shift, dims=0)
    return y

def random_crop_and_resize(x: torch.Tensor, crop_ratio: float = 0.9) -> torch.Tensor:
    """
    Randomly crop a portion and interpolate back to original size.
    Simulates different activity durations.
    """
    y = x.clone()
    
    # Handle both [C, L] and [B, C, L] shapes
    if y.ndim == 3:
        # Batch dimension present
        C, L = y.shape[1], y.shape[2]
    else:
        # No batch dimension
        C, L = y.shape[0], y.shape[1]
    
    crop_len = int(L * crop_ratio)
    
    if crop_len < L // 2:  # Safety check
        return y
    
    start = torch.randint(0, L - crop_len + 1, (1,)).item()
    
    if y.ndim == 3:
        # Process accel channel (0) for batched input
        cropped = y[:, 0, start:start + crop_len]  # [B, crop_len]
        # Interpolate: needs [B, 1, crop_len] -> [B, 1, L]
        y[:, 0, :] = torch.nn.functional.interpolate(
            cropped.unsqueeze(1),  # [B, 1, crop_len]
            size=L,
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [B, L]
    else:
        # Process accel channel (0) for non-batched input
        cropped = y[0, start:start + crop_len]
        y[0] = torch.nn.functional.interpolate(
            cropped.unsqueeze(0).unsqueeze(0),
            size=L,
            mode='linear',
            align_corners=False
        ).squeeze()
    
    return y


def make_two_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create two augmented views for contrastive learning.
    
    Consistent, moderate augmentations to stabilize training.
    Always apply the same set of augmentations to reduce variance.
    
    Args:
        x: Input tensor [B, C, L]
    
    Returns:
        Two augmented views (v1, v2)
    """
    # ALWAYS apply same augmentations for consistency
    # View 1: jitter + scale + small time shift
    v1 = x.clone()
    v1 = jitter(v1, sigma=0.05)
    v1 = scale(v1, lo=0.85, hi=1.15)
    v1 = time_shift(v1, max_shift=30)
    
    # View 2: Same augmentations with different random seeds
    v2 = x.clone()
    v2 = jitter(v2, sigma=0.05)
    v2 = scale(v2, lo=0.85, hi=1.15)
    v2 = time_shift(v2, max_shift=30)
    
    return v1, v2