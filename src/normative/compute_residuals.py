# src/accel/compute_residuals.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from .normative import load_model, predict_mu_and_residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--covars_csv", required=True)
    ap.add_argument("--model_path", required=True, help="normative_model.joblib")
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

    mu, resid, resid_norm = predict_mu_and_residuals(model, emb, df, cfg)

    np.savez_compressed(os.path.join(args.outdir, "normative_mu.npz"), mu=mu)
    np.savez_compressed(os.path.join(args.outdir, "normative_residuals.npz"), resid=resid, resid_norm=resid_norm)

    out_csv = df[[args.id_col, cfg.age_col, cfg.sex_col, cfg.accmean_col]].copy()
    out_csv["resid_norm"] = resid_norm
    out_csv.to_csv(os.path.join(args.outdir, "residual_summary.csv"), index=False)

    print("[OK] saved residuals to:", args.outdir)
    print("mu:", mu.shape, "resid:", resid.shape, "resid_norm:", resid_norm.shape)


if __name__ == "__main__":
    main()
