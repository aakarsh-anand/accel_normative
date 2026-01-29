# src/accel/embed_all.py
from __future__ import annotations

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import UKBAccelWindowDataset, WindowConfig
from .inference import load_encoder_from_ckpt, encode_windows


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collate(batch):
    # batch is list of dicts: x [K,2,L], valid [K,L], subject_id str
    x = torch.stack([b["x"] for b in batch], dim=0)         # [B,K,2,L]
    valid = torch.stack([b["valid"] for b in batch], dim=0) # [B,K,L]
    sids = [b["subject_id"] for b in batch]
    return {"x": x, "valid": valid, "subject_id": sids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config used for dataset params")
    ap.add_argument("--ckpt", required=True, help="Lightning checkpoint path")
    ap.add_argument("--outdir", required=True, help="Where to write embeddings")
    ap.add_argument("--device", default="cuda", help="cuda|cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--windows_per_subject", type=int, default=16, help="Override dataset windows_per_subject")
    ap.add_argument("--save_window_embeddings", action="store_true")
    ap.add_argument("--max_subjects", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_cfg(args.config)
    dcfg = cfg["data"]

    # Override windows_per_subject for embedding generation (more stable than training-time K)
    wcfg = WindowConfig(
        window_hours=int(dcfg["window_hours"]),
        windows_per_subject=int(args.windows_per_subject),
        min_fraction_nonmissing=float(dcfg["min_fraction_nonmissing"]),
        robust_scale=bool(dcfg["robust_scale"]),
        clip_value=float(dcfg["clip_value"]),
    )

    ds = UKBAccelWindowDataset(
        accel_dir=dcfg["accel_dir"],
        covars_csv=dcfg["covars_csv"],
        id_col=dcfg["id_col"],
        max_subjects=args.max_subjects if args.max_subjects is not None else dcfg.get("max_subjects", None),
        window_cfg=wcfg,
        seed=cfg.get("seed", 7),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    encoder = load_encoder_from_ckpt(
        args.ckpt,
        config_yaml=args.config,
        map_location="cpu"
    ).to(device)
    encoder.eval()

    all_week = []
    all_windows = [] if args.save_window_embeddings else None
    all_ids = []

    for batch in tqdm(loader, desc="Encoding", total=len(loader)):
        x = batch["x"]  # [B,K,2,L]
        sids = batch["subject_id"]

        z = encode_windows(encoder, x, device=device)  # [B,K,D]
        z = z.detach().cpu().float().numpy()

        # pooled week embedding
        z_week = z.mean(axis=1)  # [B,D]

        all_week.append(z_week)
        all_ids.extend(sids)

        if args.save_window_embeddings:
            all_windows.append(z)  # [B,K,D]

    emb_week = np.concatenate(all_week, axis=0)  # [N,D]
    pd.DataFrame({"ParticipantID": all_ids}).to_csv(os.path.join(args.outdir, "subject_ids.csv"), index=False)

    np.savez_compressed(
        os.path.join(args.outdir, "embeddings_week.npz"),
        emb=emb_week,
    )

    if args.save_window_embeddings:
        emb_win = np.concatenate(all_windows, axis=0)  # [N,K,D]
        np.savez_compressed(
            os.path.join(args.outdir, "embeddings_windows.npz"),
            emb=emb_win,
        )

    print(f"[OK] Wrote {len(all_ids)} subjects.")
    print(f"[OK] Week embeddings: {emb_week.shape}")
    if args.save_window_embeddings:
        print(f"[OK] Window embeddings: {emb_win.shape}")


if __name__ == "__main__":
    main()
