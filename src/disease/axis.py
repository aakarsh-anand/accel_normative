# src/accel/axis.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resid_dir", required=True, help="Directory with normative_residuals.npz and residual_summary.csv")
    ap.add_argument("--pd_labels_csv", required=True, help="CSV with Participant ID, Label, Status, TTE_years")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--id_col", default="Participant ID")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resid_key", default="resid", help="Key in normative_residuals.npz to use as features (e.g., resid or resid_w)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    npz_path = os.path.join(args.resid_dir, "normative_residuals.npz")
    npz = np.load(npz_path)
    if args.resid_key not in npz.files:
        raise KeyError(f"--resid_key='{args.resid_key}' not found in {npz_path}. Available keys: {npz.files}")
    resid = npz[args.resid_key].astype(np.float32)  # [N,D] or [N,K]
    subj = pd.read_csv(os.path.join(args.resid_dir, "residual_summary.csv"))
    subj_id = subj[args.id_col].astype(str)

    labels = pd.read_csv(args.pd_labels_csv)
    labels[args.id_col] = labels[args.id_col].astype(str)

    df = pd.DataFrame({args.id_col: subj_id})
    df = df.merge(labels[[args.id_col, "Label", "Status", "TTE_years"]], on=args.id_col, how="left")

    y = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int).to_numpy()
    X = resid

    # CV metrics
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    aucs, auprcs = [], []
    oof_p = np.zeros(len(y), dtype=float)

    for tr, te in skf.split(X, y):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
            )),
        ])
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof_p[te] = p
        aucs.append(roc_auc_score(y[te], p))
        auprcs.append(average_precision_score(y[te], p))

    metrics = {
        "cv_auroc_mean": float(np.mean(aucs)),
        "cv_auroc_std": float(np.std(aucs)),
        "cv_auprc_mean": float(np.mean(auprcs)),
        "cv_auprc_std": float(np.std(auprcs)),
        "n": int(len(y)),
        "n_cases": int(y.sum()),
        "n_controls": int((y == 0).sum()),
        "resid_key": str(args.resid_key),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "pd_axis_cv_metrics.csv"), index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))

    # Fit final model on all data
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])
    final_model.fit(X, y)
    joblib.dump(final_model, os.path.join(args.outdir, "pd_axis_model.joblib"))

    # PD-axis *projection score* = signed distance along coefficient vector (in standardized space)
    # We'll compute it explicitly for interpretability:
    scaler = final_model.named_steps["scaler"]
    lr = final_model.named_steps["lr"]
    Xz = scaler.transform(X)
    axis = lr.coef_.reshape(-1)  # [D]
    axis_norm = np.linalg.norm(axis) + 1e-12
    axis_unit = axis / axis_norm
    proj = (Xz @ axis_unit).astype(float)

    out = df.copy()
    out["pd_axis_prob_oof"] = oof_p
    out["pd_axis_proj"] = proj
    out.to_csv(os.path.join(args.outdir, "pd_axis_scores.csv"), index=False)

    # Save axis vector for later interpretation
    np.save(os.path.join(args.outdir, "pd_axis_unit.npy"), axis_unit)

    print("[OK] wrote pd_axis_model.joblib, pd_axis_scores.csv, pd_axis_unit.npy")


if __name__ == "__main__":
    main()
