# run_deepen.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.analysis.deepen import (
    DeepenConfig,
    load_residual_npz,
    load_residual_summary,
    load_labels,
    load_axis_scores,
    merge_subject_table,
    plot_offramp_umap,
    plot_leadtime_heatmap,
    find_mislabeled_cohorts,
    write_deepening_outputs,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait_outdir", required=True, help="outputs/traits/<trait>/")
    ap.add_argument("--resid_dir", required=True, help="outputs/residuals_*/")
    ap.add_argument("--axis_scores_csv", required=True, help="axis scores csv (trait-specific)")
    ap.add_argument("--labels_csv", required=True, help="labels.csv for trait")
    ap.add_argument("--resid_key", default="resid", help="Which residual vectors to use for manifold (resid or resid_w)")
    ap.add_argument("--score_col", default=None, help="Column name in axis scores csv to use as score")
    args = ap.parse_args()

    cfg = DeepenConfig()

    trait_outdir = Path(args.trait_outdir)
    outdir = trait_outdir / "deepening"
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)

    resid_summary = load_residual_summary(args.resid_dir)
    labels = load_labels(args.labels_csv)
    scores = load_axis_scores(args.axis_scores_csv)

    # Normalize score column naming
    if args.score_col is None:
        # try common defaults
        for c in ["pd_axis", "axis_score", "score", "proj", "logit"]:
            if c in scores.columns:
                args.score_col = c
                break
    if args.score_col is None:
        raise ValueError(f"Could not infer score_col from columns: {list(scores.columns)}")

    if args.score_col != cfg.score_col:
        scores = scores.rename(columns={args.score_col: cfg.score_col})

    df = merge_subject_table(resid_summary, labels, scores, cfg)

    # Off-ramp manifold
    npz = load_residual_npz(args.resid_dir)
    if args.resid_key not in npz:
        raise KeyError(f"{args.resid_key} not in normative_residuals.npz; available: {list(npz.keys())}")
    X = npz[args.resid_key]
    # Align with df order using Participant ID from residual_summary
    # Assumes residual_summary.csv rows align with npz order in compute_residuals (your current behavior).
    plot_offramp_umap(X=X, df=df, out_png=outdir / "figs" / "offramp_umap.png", cfg=cfg, color_by=cfg.status_col)

    # Lead-time heatmap
    plot_leadtime_heatmap(df=df, out_png=outdir / "figs" / "leadtime_heatmap.png", cfg=cfg, value_col=cfg.score_col)

    # Mislabeled analysis table (cohort flags)
    df2 = find_mislabeled_cohorts(df, cfg)
    df2.to_csv(outdir / "tables" / "mislabeled_cohorts.csv", index=False)

    write_deepening_outputs(df=df2, outdir=outdir, cfg=cfg, notes={"resid_key": args.resid_key})

    print("[OK] wrote:", outdir)

if __name__ == "__main__":
    main()
