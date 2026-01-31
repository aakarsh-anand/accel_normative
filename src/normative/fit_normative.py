# src/normative/fit_normative.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from .normative import NormativeConfig, fit_normative_model, save_model, fit_residual_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True, help="Directory with embeddings_week.npz and subject_ids.csv")
    ap.add_argument("--covars_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--id_col", default="Participant ID")
    ap.add_argument("--age_col", default="Age")
    ap.add_argument("--sex_col", default="Sex")
    ap.add_argument("--accmean_col", default="Overall acceleration average")

    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--n_knots", type=int, default=8)
    ap.add_argument("--degree", type=int, default=3)

    ap.add_argument("--ref_ids_csv", default=None, help="Optional CSV with a column of Participant IDs to fit normative model on")
    ap.add_argument("--ref_id_col", default=None, help="ID column name in ref_ids_csv (default: same as --id_col)")
    ap.add_argument("--geom_var", type=float, default=0.90, help="PCA variance to retain for residual geometry (Phase 1)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    emb = np.load(os.path.join(args.emb_dir, "embeddings_week.npz"))["emb"]  # [N,D]
    ids = pd.read_csv(os.path.join(args.emb_dir, "subject_ids.csv"))["ParticipantID"].astype(str)

    cov = pd.read_csv(args.covars_csv)
    cov[args.id_col] = cov[args.id_col].astype(str)

    df = pd.DataFrame({args.id_col: ids})
    df = df.merge(cov, on=args.id_col, how="left")

    # Build reference mask if provided
    ref_mask = None
    if args.ref_ids_csv is not None:
        ref = pd.read_csv(args.ref_ids_csv)
        ref_id_col = args.ref_id_col or args.id_col
        ref_ids = set(ref[ref_id_col].astype(str).tolist())
        ref_mask = df[args.id_col].astype(str).isin(ref_ids).to_numpy()
        print(f"[Ref] Using {ref_mask.sum()} / {len(ref_mask)} subjects for normative fit")

    cfg = NormativeConfig(
        id_col=args.id_col,
        age_col=args.age_col,
        sex_col=args.sex_col,
        accmean_col=args.accmean_col,
        n_knots=args.n_knots,
        degree=args.degree,
        alpha=args.alpha,
    )

    model, metrics = fit_normative_model(emb=emb, covars=df, cfg=cfg, ref_mask=ref_mask)

    # --- Phase 1: fit residual geometry on the SAME reference set used for fitting ---
    # This does not affect the mean model; it defines a statistically meaningful metric
    # for deviation magnitude (and optionally whitened residual vectors).
    Xdf = df[[cfg.age_col, cfg.sex_col, cfg.accmean_col]].copy()
    if ref_mask is None:
        ref_mask_geom = np.ones(len(df), dtype=bool)
    else:
        ref_mask_geom = ref_mask

    mu_fit = model.predict(Xdf.loc[ref_mask_geom])
    resid_fit = emb[ref_mask_geom] - mu_fit
    geometry = fit_residual_geometry(resid_fit=resid_fit, var_explained=args.geom_var)

    geom_path = os.path.join(args.outdir, "geometry.joblib")
    joblib.dump(geometry, geom_path)
    print("[OK] saved geometry:", geom_path)


    out_model = os.path.join(args.outdir, "normative_model.joblib")
    save_model(model, cfg, out_model)

    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "normative_fit_metrics.csv"), index=False)
    print("[OK] saved:", out_model)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
