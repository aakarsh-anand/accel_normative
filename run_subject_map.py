# run_subject_map.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.analysis.viz import savefig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resid_dir", required=True)
    ap.add_argument("--traits_yaml", default="configs/traits.yaml")
    ap.add_argument("--outdir", default="outputs/subject_maps")
    ap.add_argument("--resid_key", default="resid")
    ap.add_argument("--id_col", default="Participant ID")
    ap.add_argument("--max_controls", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load residuals + ids
    npz = np.load(Path(args.resid_dir) / "normative_residuals.npz")
    X = npz[args.resid_key].astype(np.float32)
    summ = pd.read_csv(Path(args.resid_dir) / "residual_summary.csv")
    ids = summ[args.id_col].astype(str).values

    # Fit a single global UMAP on a manageable subset (controls-heavy datasets need subsampling)
    rng = np.random.default_rng(args.seed)
    idx_all = np.arange(len(ids))

    # If you have labels for a trait, we’ll highlight. For embedding, we just subsample uniformly.
    if len(idx_all) > args.max_controls:
        idx_fit = rng.choice(idx_all, size=args.max_controls, replace=False)
    else:
        idx_fit = idx_all

    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=args.seed, metric="euclidean")
    emb2_fit = reducer.fit_transform(X[idx_fit])

    # Transform all points (UMAP supports transform)
    try:
        emb2 = reducer.transform(X)
    except Exception:
        # fallback: only output fit subset
        emb2 = np.full((len(ids), 2), np.nan, dtype=float)
        emb2[idx_fit] = emb2_fit

    # Base plot (all points)
    plt.figure(figsize=(7.0, 6.0))
    plt.scatter(emb2[idx_fit, 0], emb2[idx_fit, 1], s=2, alpha=0.25)
    plt.title(f"Subject residual map (UMAP) — subset n={len(idx_fit)}")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    savefig(outdir / "subjects_umap_base.png")

    # Trait overlays
    with open(args.traits_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    for name in cfg.get("traits", []):
        # name = t["name"]
        lab_path = Path("outputs/traits") / name / "labels.csv"
        if not lab_path.exists():
            continue
        lab = pd.read_csv(lab_path)
        if "Status" not in lab.columns:
            continue

        # align labels to ids
        lab = lab[[args.id_col, "Status"]].copy()
        lab[args.id_col] = lab[args.id_col].astype(str)
        lab = lab.set_index(args.id_col).reindex(ids)

        status = lab["Status"].astype(str).fillna("NA").values

        plt.figure(figsize=(7.0, 6.0))
        # background
        plt.scatter(emb2[idx_fit, 0], emb2[idx_fit, 1], s=2, alpha=0.08)

        # overlay statuses
        for s, a, size in [("Control", 0.35, 8), ("Prodromal", 0.85, 18), ("Diagnosed", 0.85, 18)]:
            m = status == s
            if m.any():
                plt.scatter(emb2[m, 0], emb2[m, 1], s=size, alpha=a, label=s)

        plt.legend(frameon=False)
        plt.title(f"{name}: cases overlaid on subject UMAP")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        savefig(outdir / f"subjects_umap_{name}.png")

    print("[OK] wrote subject maps to", outdir)

if __name__ == "__main__":
    main()
