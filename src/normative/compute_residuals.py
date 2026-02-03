# src/accel/compute_residuals.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from .normative import load_model, predict_mu_and_residuals, whiten_residuals, mahalanobis_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--covars_csv", required=True)
    ap.add_argument("--wear_dates_csv", default=None, help="Optional wear dates CSV (needed if model used seasonal covariates)")
    ap.add_argument("--model_path", required=True, help="normative_model.joblib")
    ap.add_argument("--geometry_path", default=None, help="Optional geometry.joblib")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--id_col", default="Participant ID")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load embeddings
    emb = np.load(os.path.join(args.emb_dir, "embeddings_week.npz"))["emb"]
    ids = pd.read_csv(os.path.join(args.emb_dir, "subject_ids.csv"))["ParticipantID"].astype(str)

    # Load covariates
    cov = pd.read_csv(args.covars_csv)
    cov[args.id_col] = cov[args.id_col].astype(str)

    df = pd.DataFrame({args.id_col: ids})
    df = df.merge(cov, on=args.id_col, how="left")

    # Load model and config
    model, cfg = load_model(args.model_path)
    
    # Check if model uses seasonal covariates
    if cfg.month_sin_col is not None or cfg.month_cos_col is not None:
        if args.wear_dates_csv is None:
            print("[WARNING] Model was trained with seasonal covariates but --wear_dates_csv not provided!")
            print("          This will likely cause errors. Please provide --wear_dates_csv")
        else:
            wear_dates = pd.read_csv(args.wear_dates_csv)
            wear_dates[args.id_col] = wear_dates[args.id_col].astype(str)
            df = df.merge(wear_dates[[args.id_col, 'wear_month_sin', 'wear_month_cos']], 
                         on=args.id_col, how="left")
            print(f"[Season] Loaded seasonal covariates from {args.wear_dates_csv}")

    # Load geometry
    geometry = None
    geom_path = args.geometry_path
    if geom_path is None:
        cand = os.path.join(os.path.dirname(args.model_path), "geometry.joblib")
        if os.path.exists(cand):
            geom_path = cand
    if geom_path is not None and os.path.exists(geom_path):
        geometry = joblib.load(geom_path)
        print(f"[Geom] Loaded geometry from {geom_path}")
    else:
        print("[Geom] No geometry found; will only compute Euclidean resid_norm")

    # Compute residuals
    print(f"[Predict] Computing residuals...")
    if isinstance(model, list):
        print(f"[Predict] Using {len(model)} per-dimension models")
    else:
        print(f"[Predict] Using multi-output model")
    
    mu, resid, resid_norm = predict_mu_and_residuals(model, emb, df, cfg)

    # Whiten if geometry available
    resid_w = None
    resid_norm_mahal = None
    if geometry is not None:
        print(f"[Whiten] Whitening residuals...")
        resid_w = whiten_residuals(resid, geometry)   # [N,K]
        resid_norm_mahal = mahalanobis_norm(resid_w)  # [N]

    # Save outputs
    np.savez_compressed(os.path.join(args.outdir, "normative_mu.npz"), mu=mu)
    
    payload = {"resid": resid, "resid_norm": resid_norm}
    if resid_w is not None:
        payload["resid_w"] = resid_w
    if resid_norm_mahal is not None:
        payload["resid_norm_mahal"] = resid_norm_mahal
    np.savez_compressed(os.path.join(args.outdir, "normative_residuals.npz"), **payload)

    # Create summary CSV
    from .normative import _get_covariate_columns
    covar_cols = _get_covariate_columns(cfg)
    out_csv = df[[args.id_col] + covar_cols].copy()
    out_csv["resid_norm"] = resid_norm
    if resid_norm_mahal is not None:
        out_csv["resid_norm_mahal"] = resid_norm_mahal
    out_csv.to_csv(os.path.join(args.outdir, "residual_summary.csv"), index=False)

    print("\n" + "="*60)
    print("RESIDUALS COMPUTED")
    print("="*60)
    print(f"mu shape:        {mu.shape}")
    print(f"resid shape:     {resid.shape}")
    print(f"resid_norm:      {resid_norm.shape} (mean={resid_norm.mean():.3f}, std={resid_norm.std():.3f})")
    if resid_w is not None:
        print(f"resid_w shape:   {resid_w.shape}")
        print(f"resid_norm_mahal: {resid_norm_mahal.shape} (mean={resid_norm_mahal.mean():.3f}, std={resid_norm_mahal.std():.3f})")
    print(f"\nOutputs saved to: {args.outdir}")
    print("="*60)


if __name__ == "__main__":
    main()