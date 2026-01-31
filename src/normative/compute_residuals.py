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
    ap.add_argument("--model_path", required=True, help="normative_model.joblib")
    ap.add_argument("--geometry_path", default=None, help="Optional geometry.joblib (defaults to alongside model_path if present)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--id_col", default="Participant ID")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    emb = np.load(os.path.join(args.emb_dir, "embeddings_week.npz"))["emb"]
    ids = pd.read_csv(os.path.join(args.emb_dir, "subject_ids.csv"))["ParticipantID"].astype(str)

    cov = pd.read_csv(args.covars_csv)
    cov[args.id_col] = cov[args.id_col].astype(str)

    df = pd.DataFrame({args.id_col: ids})
    df = df.merge(cov, on=args.id_col, how="left")

    model, cfg = load_model(args.model_path)

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

    mu, resid, resid_norm = predict_mu_and_residuals(model, emb, df, cfg)

    resid_w = None
    resid_norm_mahal = None
    if geometry is not None:
        resid_w = whiten_residuals(resid, geometry)   # [N,K]
        resid_norm_mahal = mahalanobis_norm(resid_w)  # [N]

    np.savez_compressed(os.path.join(args.outdir, "normative_mu.npz"), mu=mu)
    payload = {"resid": resid, "resid_norm": resid_norm}
    if resid_w is not None:
        payload["resid_w"] = resid_w
    if resid_norm_mahal is not None:
        payload["resid_norm_mahal"] = resid_norm_mahal
    np.savez_compressed(os.path.join(args.outdir, "normative_residuals.npz"), **payload)

    out_csv = df[[args.id_col, cfg.age_col, cfg.sex_col, cfg.accmean_col]].copy()
    out_csv["resid_norm"] = resid_norm
    if resid_norm_mahal is not None:
        out_csv["resid_norm_mahal"] = resid_norm_mahal
    out_csv.to_csv(os.path.join(args.outdir, "residual_summary.csv"), index=False)

    print("[OK] saved residuals to:", args.outdir)
    print("mu:", mu.shape, "resid:", resid.shape, "resid_norm:", resid_norm.shape)
    if resid_w is not None:
        print("resid_w:", resid_w.shape, "resid_norm_mahal:", resid_norm_mahal.shape)


if __name__ == "__main__":
    main()
