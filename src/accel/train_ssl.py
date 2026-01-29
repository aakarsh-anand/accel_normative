from __future__ import annotations
import os
import argparse
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import WandbLogger
except Exception:
    WandbLogger = None

from torch.utils.data import DataLoader

from .data import UKBAccelWindowDataset, WindowConfig
from .models import EncoderConfig
from .lightning_module import SSLMAEBYOLModule, SSLConfig


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    pl.seed_everything(cfg.get("seed", 7), workers=True)

    dcfg = cfg["data"]
    wcfg = WindowConfig(
        window_hours=int(dcfg["window_hours"]),
        windows_per_subject=int(dcfg["windows_per_subject"]),
        min_fraction_nonmissing=float(dcfg["min_fraction_nonmissing"]),
        robust_scale=bool(dcfg["robust_scale"]),
        clip_value=float(dcfg["clip_value"]),
    )

    ds = UKBAccelWindowDataset(
        accel_dir=dcfg["accel_dir"],
        covars_csv=dcfg["covars_csv"],
        id_col=dcfg["id_col"],
        max_subjects=dcfg.get("max_subjects", None),
        window_cfg=wcfg,
        seed=cfg.get("seed", 7),
    )

    # Simple split for now (SSL doesn't truly need val; but useful for sanity)
    n = len(ds)
    n_val = max(1, int(0.01 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    def collate(batch):
        # batch list of dicts with x [K,2,L], valid [K,L]
        # stack -> [B,K,2,L]
        import torch
        x = torch.stack([b["x"] for b in batch], dim=0)
        valid = torch.stack([b["valid"] for b in batch], dim=0)
        return {"x": x, "valid": valid}

    train_loader = DataLoader(
        train_ds,
        batch_size=int(dcfg["batch_size"]),
        shuffle=True,
        num_workers=int(dcfg["num_workers"]),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(dcfg["batch_size"]),
        shuffle=False,
        num_workers=int(dcfg["num_workers"]) // 8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate,
    )

    L = int((wcfg.window_hours * 3600) / 5)

    enc_cfg = EncoderConfig(
        in_channels=2,
        base_channels=int(cfg["model"]["base_channels"]),
        depth=int(cfg["model"]["depth"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
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

    mod = SSLMAEBYOLModule(
        encoder_cfg=enc_cfg,
        ssl_cfg=ssl_cfg,
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
        warmup_steps=int(cfg["optim"]["warmup_steps"]),
        max_steps=int(cfg["optim"]["max_steps"]),
        grad_clip=float(cfg["optim"]["grad_clip"]),
        window_len=L,
        ema_decay_base=float(cfg["optim"]["ema_decay_base"]),
        ema_warmup_steps=float(cfg["optim"]["ema_warmup_steps"])
    )

    loggers = []
    loggers.append(CSVLogger(save_dir="logs", name="ssl"))

    lcfg = cfg.get("logging", {})
    if lcfg.get("use_wandb", False):
        if WandbLogger is None:
            raise RuntimeError("WandbLogger not available. Install wandb.")
        loggers.append(WandbLogger(
            project=lcfg.get("wandb_project", "accel-normative"),
            entity=lcfg.get("wandb_entity", None),
            name=lcfg.get("run_name", None),
            log_model=False,
        ))

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join("checkpoints", "ssl"),
            filename="step{step}-loss{train/loss:.3f}",
            save_top_k=3,
            monitor="train/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    tcfg = cfg["trainer"]
    steps_per_epoch = len(train_loader)
    max_steps = int(tcfg.get("max_steps", 100000))
    max_epochs = (max_steps // steps_per_epoch) + 1
    trainer = pl.Trainer(
        accelerator=tcfg.get("accelerator", "gpu"),
        devices=int(tcfg.get("devices", 1)),
        strategy=tcfg.get("strategy", "ddp"),
        precision=tcfg.get("precision", "16-mixed"),
        max_epochs=max_epochs,
        max_steps=max_steps,
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 50)),
        val_check_interval=int(tcfg.get("val_check_interval", 2000)),
        limit_val_batches=tcfg.get("limit_val_batches", 10),
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=float(cfg["optim"]["grad_clip"]),
        benchmark=True,
    )

    trainer.fit(mod, train_loader, val_loader)


if __name__ == "__main__":
    main()
