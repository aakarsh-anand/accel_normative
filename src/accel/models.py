from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=7, stride=stride, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(32, c_out), num_channels=c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=7, stride=1, padding=3, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(32, c_out), num_channels=c_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.proj = None
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(32, c_out), num_channels=c_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.silu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.gn2(out)

        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        out = F.silu(out)
        return out


@dataclass
class EncoderConfig:
    in_channels: int = 2
    base_channels: int = 64
    depth: int = 4
    embed_dim: int = 256
    dropout: float = 0.0


class ResNet1DEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        c = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.in_channels, c, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(num_groups=min(32, c), num_channels=c),
            nn.SiLU(),
        )
        blocks = []
        in_c = c
        for i in range(cfg.depth):
            out_c = c * (2 ** i)
            stride = 2 if i > 0 else 1
            blocks.append(ResidualBlock1D(in_c, out_c, stride=stride, dropout=cfg.dropout))
            blocks.append(ResidualBlock1D(out_c, out_c, stride=1, dropout=cfg.dropout))
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_c, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        h = self.stem(x)
        h = self.blocks(h)
        z = self.head(h)
        return z  # [B, D]


class MAEDecoder1D(nn.Module):
    """Small decoder predicting accel channel only."""
    def __init__(self, embed_dim: int, out_len: int, hidden: int = 256):
        super().__init__()
        self.out_len = out_len
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D] -> [B, L]
        return self.net(z)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1)  # [B]
