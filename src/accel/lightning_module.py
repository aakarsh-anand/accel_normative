from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .models import ResNet1DEncoder, MAEDecoder1D, MLP, byol_loss, EncoderConfig
from .transforms import SpanMasker, MaskConfig, make_two_views


@dataclass
class SSLConfig:
    mask_ratio: float = 0.6
    mask_span_min: int = 10
    mask_span_max: int = 200
    recon_loss: str = "huber"     # "l1" | "mse" | "huber"
    recon_delta: float = 1.0
    byol_weight: float = 0.2
    proj_dim: int = 256
    pred_dim: int = 256


class SSLMAEBYOLModule(pl.LightningModule):
    def __init__(
        self,
        encoder_cfg: EncoderConfig,
        ssl_cfg: SSLConfig,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        grad_clip: float = 1.0,
        window_len: int = 8640,
        ema_decay_base: float = 0.996,
        ema_warmup_steps: int = 2000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_cfg", "ssl_cfg"])

        # Online networks (trainable)
        self.encoder = ResNet1DEncoder(encoder_cfg)
        self.decoder = MAEDecoder1D(embed_dim=encoder_cfg.embed_dim, out_len=window_len)
        self.projector = MLP(encoder_cfg.embed_dim, ssl_cfg.proj_dim, hidden=512)
        self.predictor = MLP(ssl_cfg.proj_dim, ssl_cfg.pred_dim, hidden=512)

        # Target networks (updated via EMA, not gradients)
        self.target_encoder = ResNet1DEncoder(encoder_cfg)
        self.target_projector = MLP(encoder_cfg.embed_dim, ssl_cfg.proj_dim, hidden=512)
        
        # Initialize target networks with same weights as online networks
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_projector.load_state_dict(self.projector.state_dict())
        
        # Freeze target networks (no gradient updates)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        self.target_encoder.eval()
        self.target_projector.eval()

        self.ssl_cfg = ssl_cfg
        self.ema_decay_base = ema_decay_base
        self.ema_warmup_steps = ema_warmup_steps
        self.masker = SpanMasker(MaskConfig(
            mask_ratio=ssl_cfg.mask_ratio,
            span_min=ssl_cfg.mask_span_min,
            span_max=ssl_cfg.mask_span_max,
            seed=7
        ))

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.window_len = window_len

    def _get_ema_decay(self):
        """Get current EMA decay rate with warmup schedule"""
        step = self.global_step
        if step < self.ema_warmup_steps:
            # Start with lower decay (faster updates) and gradually increase
            progress = step / self.ema_warmup_steps
            return progress * self.ema_decay_base
        return self.ema_decay_base

    @torch.no_grad()
    def _ema_update_module(self, online: nn.Module, target: nn.Module, decay: float):
        o = online.state_dict()
        t = target.state_dict()
        for k in t.keys():
            if t[k].dtype.is_floating_point:
                t[k].mul_(decay).add_(o[k], alpha=1.0 - decay)
            else:
                t[k].copy_(o[k])  # e.g. num_batches_tracked
        target.load_state_dict(t, strict=True)

    @torch.no_grad()
    def _update_target_network(self):
        d = self._get_ema_decay()
        self._ema_update_module(self.encoder, self.target_encoder, d)      # harmless (GN)
        self._ema_update_module(self.projector, self.target_projector, d)  # important (BN)

    def _recon_loss(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # pred/target: [B, L], weight: [B, L] (1 means include in loss)
        if self.ssl_cfg.recon_loss == "l1":
            err = (pred - target).abs()
        elif self.ssl_cfg.recon_loss == "mse":
            err = (pred - target) ** 2
        else:  # huber
            err = F.huber_loss(pred, target, delta=self.ssl_cfg.recon_delta, reduction="none")
        loss = (err * weight).sum() / weight.sum().clamp_min(1.0)
        return loss
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # batch["x"]: [B, K, 2, L], batch["valid"]: [B, K, L]
        x = batch["x"]
        valid = batch["valid"]
        B, K, C, L = x.shape

        # Flatten windows into batch dimension
        x = x.view(B * K, C, L)
        valid = valid.view(B * K, L)

        # Two views for BYOL
        v1, v2 = make_two_views(x)

        # MAE masking: keep mask is bool [L]; True=visible to encoder
        keep1 = torch.stack([self.masker(L) for _ in range(B * K)], dim=0).to(x.device)  # [BK, L]
        keep2 = torch.stack([self.masker(L) for _ in range(B * K)], dim=0).to(x.device)

        v1_masked = v1.clone()
        v2_masked = v2.clone()
        # Mask accel channel (0) only in the input
        # Apply different masks to each view
        keep1_expanded = keep1.unsqueeze(1).float()  # [BK, 1, L]
        keep2_expanded = keep2.unsqueeze(1).float()  # [BK, 1, L]
        v1_masked[:, 0:1, :] *= keep1_expanded
        v2_masked[:, 0:1, :] *= keep2_expanded

        # Encode with online network (NO L2 normalization)
        z1 = self.encoder(v1_masked)
        z2 = self.encoder(v2_masked)

        # Track representation collapse metrics
        with torch.no_grad():
            z_std = 0.5 * (z1.std(dim=0).mean() + z2.std(dim=0).mean())
            z_mean = 0.5 * (z1.mean(dim=0).abs().mean() + z2.mean(dim=0).abs().mean())
            z_sim = F.cosine_similarity(z1, z2, dim=-1).mean()

        # Reconstruction from z (predict accel channel)
        rec1 = self.decoder(z1)  # [BK, L]
        rec2 = self.decoder(z2)

        target1 = v1[:, 0, :]    # accel only
        target2 = v2[:, 0, :]

        # Loss only on masked positions AND valid points
        weight1 = (~keep1).float() * valid
        weight2 = (~keep2).float() * valid
        recon_loss = 0.5 * (self._recon_loss(rec1, target1, weight1) + self._recon_loss(rec2, target2, weight2))

        # BYOL loss with EMA target network
        # Online network: project embeddings, then predict
        p1 = self.predictor(self.projector(z1))
        p2 = self.predictor(self.projector(z2))
        
        # Target network projections (no predictor)
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(v1_masked))
            t2 = self.target_projector(self.target_encoder(v2_masked))

        byol = 0.5 * (byol_loss(p1, t2).mean() + byol_loss(p2, t1).mean())

        loss = recon_loss + self.ssl_cfg.byol_weight * byol

        # Update target network after computing loss
        self._update_target_network()

        with torch.no_grad():
            p1n = F.normalize(p1, dim=-1)
            p2n = F.normalize(p2, dim=-1)
            t1n = F.normalize(t1, dim=-1)
            t2n = F.normalize(t2, dim=-1)

            p_std = 0.5 * (p1n.std(dim=0).mean() + p2n.std(dim=0).mean())
            t_std = 0.5 * (t1n.std(dim=0).mean() + t2n.std(dim=0).mean())
            pt_sim = 0.5 * (
                F.cosine_similarity(p1n, t2n, dim=-1).mean() +
                F.cosine_similarity(p2n, t1n, dim=-1).mean()
            )

        # Logging
        self.log("train/recon_loss", recon_loss, prog_bar=True, on_step=True)
        self.log("train/byol_loss", byol, prog_bar=False, on_step=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        # Log collapse metrics
        self.log("train/z_std", z_std, prog_bar=True, on_step=True)
        self.log("train/z_mean", z_mean, prog_bar=False, on_step=True)
        self.log("train/z_sim", z_sim, prog_bar=False, on_step=True)
        self.log("train/p_std", p_std, on_step=True)
        self.log("train/t_std", t_std, on_step=True)
        self.log("train/pt_sim", pt_sim, on_step=True)
        self.log("train/ema_decay", self._get_ema_decay(), prog_bar=False, on_step=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # batch["x"]: [B, K, 2, L], batch["valid"]: [B, K, L]
        x = batch["x"]
        valid = batch["valid"]
        B, K, C, L = x.shape

        # Flatten windows into batch dimension
        x = x.view(B * K, C, L)
        valid = valid.view(B * K, L)

        # Two views for BYOL
        v1, v2 = make_two_views(x)

        # MAE masking: keep mask is bool [L]; True=visible to encoder
        keep1 = torch.stack([self.masker(L) for _ in range(B * K)], dim=0).to(x.device)  # [BK, L]
        keep2 = torch.stack([self.masker(L) for _ in range(B * K)], dim=0).to(x.device)

        v1_masked = v1.clone()
        v2_masked = v2.clone()
        # Mask accel channel (0) only in the input
        # Apply different masks to each view
        keep1_expanded = keep1.unsqueeze(1).float()  # [BK, 1, L]
        keep2_expanded = keep2.unsqueeze(1).float()  # [BK, 1, L]
        v1_masked[:, 0:1, :] *= keep1_expanded
        v2_masked[:, 0:1, :] *= keep2_expanded

        # Encode with online network (NO L2 normalization)
        z1 = self.encoder(v1_masked)
        z2 = self.encoder(v2_masked)

        # Track representation collapse metrics
        with torch.no_grad():
            z_std = 0.5 * (z1.std(dim=0).mean() + z2.std(dim=0).mean())
            z_mean = 0.5 * (z1.mean(dim=0).abs().mean() + z2.mean(dim=0).abs().mean())
            z_sim = F.cosine_similarity(z1, z2, dim=-1).mean()

        # Reconstruction from z (predict accel channel)
        rec1 = self.decoder(z1)  # [BK, L]
        rec2 = self.decoder(z2)

        target1 = v1[:, 0, :]    # accel only
        target2 = v2[:, 0, :]

        # Loss only on masked positions AND valid points
        weight1 = (~keep1).float() * valid
        weight2 = (~keep2).float() * valid
        recon_loss = 0.5 * (self._recon_loss(rec1, target1, weight1) + self._recon_loss(rec2, target2, weight2))

        # BYOL loss with EMA target network
        p1 = self.predictor(self.projector(z1))
        p2 = self.predictor(self.projector(z2))
        
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(v1_masked))
            t2 = self.target_projector(self.target_encoder(v2_masked))

        byol = 0.5 * (byol_loss(p1, t2).mean() + byol_loss(p2, t1).mean())

        loss = recon_loss + self.ssl_cfg.byol_weight * byol

        with torch.no_grad():
            p1n = F.normalize(p1, dim=-1)
            p2n = F.normalize(p2, dim=-1)
            t1n = F.normalize(t1, dim=-1)
            t2n = F.normalize(t2, dim=-1)

            p_std = 0.5 * (p1n.std(dim=0).mean() + p2n.std(dim=0).mean())
            t_std = 0.5 * (t1n.std(dim=0).mean() + t2n.std(dim=0).mean())
            pt_sim = 0.5 * (
                F.cosine_similarity(p1n, t2n, dim=-1).mean() +
                F.cosine_similarity(p2n, t1n, dim=-1).mean()
            )

        # Log with val prefix
        self.log("val/recon_loss", recon_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/byol_loss", byol, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        # Log collapse metrics
        self.log("val/z_std", z_std, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/z_mean", z_mean, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/z_sim", z_sim, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/p_std", p_std, on_step=True, sync_dist=True)
        self.log("val/t_std", t_std, on_step=True, sync_dist=True)
        self.log("val/pt_sim", pt_sim, on_step=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Only optimize online networks (target networks update via EMA)
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def lr_lambda(step: int):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            # cosine decay to 0.1
            progress = (step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}