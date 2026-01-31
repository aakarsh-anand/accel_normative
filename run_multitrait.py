# run_multitrait.py
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from src.analysis.multitrait import (
    trait_embedding_from_axes,
    trait_embedding_from_centroids,
    plot_trait_umap,
    cluster_traits,
)
from src.analysis.icd_denoise import knn_enrichment, candidate_relabels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits_yaml", default="configs/traits.yaml")
    ap.add_argument("--resid_dir", required=True)
    ap.add_argument("--outdir", default="outputs/multitrait")
    ap.add_argument("--resid_key", default="resid", help="resid or resid_w")
    ap.add_argument("--use", choices=["axes", "centroids"], default="axes")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--id_col", default="Participant ID")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.traits_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    traits = []
    # Expect cfg structure similar to your run_traits usage: list of traits with names and output dirs
    for name in cfg.get("traits", []):
        # name = t["name"]
        trait_out = Path("outputs/traits") / name
        traits.append({
            "name": name,
            "labels_csv": str(trait_out / "labels.csv"),
            "axis_dir": str(trait_out / "axis"),
        })

    # Load residual vectors
    npz = np.load(Path(args.resid_dir) / "normative_residuals.npz")
    if args.resid_key not in npz.files:
        raise KeyError(f"{args.resid_key} not in residual npz; available: {npz.files}")
    X = npz[args.resid_key].astype(np.float32)
    resid_summary = pd.read_csv(Path(args.resid_dir) / "residual_summary.csv")
    ids = resid_summary[args.id_col].astype(str).values

    # Trait embedding + atlas
    if args.use == "axes":
        Xt, names = trait_embedding_from_axes(traits)
    else:
        Xt, names = trait_embedding_from_centroids(X, resid_summary[args.id_col], traits, id_col=args.id_col)

    plot_trait_umap(Xt, names, outdir / "trait_umap.png")
    cl = cluster_traits(Xt)
    pd.DataFrame({"trait": names, "cluster": cl}).to_csv(outdir / "trait_clusters.csv", index=False)

    # ICD denoise primitives
    den_dir = outdir / "icd_denoise"
    den_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in traits:
        lab_path = Path(t["labels_csv"])
        if not lab_path.exists():
            continue
        lab = pd.read_csv(lab_path)
        if "Label" not in lab.columns:
            continue
        enrich = knn_enrichment(X, ids, lab, id_col=args.id_col, label_col="Label", k=args.k)
        cand = candidate_relabels(enrich)
        cand.to_csv(den_dir / f"{t['name']}_candidates.csv", index=False)

        rows.append({
            "trait": t["name"],
            "n": int(len(cand)),
            "n_candidate_fp": int(cand["candidate_fp"].sum()),
            "n_candidate_fn": int(cand["candidate_fn"].sum()),
        })

    pd.DataFrame(rows).to_csv(outdir / "trait_table.csv", index=False)
    print("[OK] wrote multi-trait outputs to", outdir)

if __name__ == "__main__":
    main()
