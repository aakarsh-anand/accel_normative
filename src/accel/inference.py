# src/accel/inference.py
from __future__ import annotations
from typing import Optional
import yaml
import torch

from .models import EncoderConfig
from .lightning_module import SSLMAEBYOLModule, SSLConfig


def load_encoder_from_ckpt(
    ckpt_path: str,
    config_yaml: str,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """
    Load Lightning checkpoint + rebuild model configs properly.
    Returns encoder in eval mode.
    """

    # Load YAML config used during training
    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    # Rebuild encoder config
    enc_cfg = EncoderConfig(
        in_channels=2,
        base_channels=int(cfg["model"]["base_channels"]),
        depth=int(cfg["model"]["depth"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )

    # Rebuild SSL config
    ssl_cfg = SSLConfig(
        mask_ratio=float(cfg["ssl"]["mask_ratio"]),
        mask_span_min=int(cfg["ssl"]["mask_span_min"]),
        mask_span_max=int(cfg["ssl"]["mask_span_max"]),
        recon_loss=str(cfg["ssl"]["recon_loss"]),
        recon_delta=float(cfg["ssl"]["recon_delta"]),
        byol_weight=float(cfg["ssl"]["byol_weight"]),
        proj_dim=int(cfg["ssl"]["proj_dim"]),
        pred_dim=int(cfg["ssl"]["pred_dim"]),
    )

    # Window length must match training
    wcfg = cfg["data"]
    L = int((int(wcfg["window_hours"]) * 3600) / 5)

    # Now load module correctly
    mod = SSLMAEBYOLModule.load_from_checkpoint(
        ckpt_path,
        encoder_cfg=enc_cfg,
        ssl_cfg=ssl_cfg,
        window_len=L,
        map_location=map_location,
    )

    enc = mod.encoder
    enc.eval()
    return enc


@torch.no_grad()
def encode_windows(
    encoder: torch.nn.Module,
    x: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Encode windows.
    x: [B, K, C, L] or [B, C, L]
    returns:
      if input is [B,K,C,L] -> [B,K,D]
      if input is [B,C,L]   -> [B,D]
    """
    if device is not None:
        encoder = encoder.to(device)
        x = x.to(device)

    if x.dim() == 4:
        B, K, C, L = x.shape
        z = encoder(x.view(B * K, C, L)).view(B, K, -1)
        return z
    elif x.dim() == 3:
        return encoder(x)
    else:
        raise ValueError(f"Expected x dim 3 or 4, got {x.shape}")
