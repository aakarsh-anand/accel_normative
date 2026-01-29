# src/accel/residual_probes.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, accuracy_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resid_dir", required=True, help="Directory with normative_residuals.npz and residual_summary.csv")
    ap.add_argument("--id_col", default="Participant ID")
    ap.add_argument("--age_col", default="Age")
    ap.add_argument("--sex_col", default="Sex")
    ap.add_argument("--accmean_col", default="Overall acceleration average")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    resid = np.load(os.path.join(args.resid_dir, "normative_residuals.npz"))["resid"]  # [N,D]
    resid_norm = np.load(os.path.join(args.resid_dir, "normative_residuals.npz"))["resid_norm"]  # [N]
    df = pd.read_csv(os.path.join(args.resid_dir, "residual_summary.csv"))

    X = resid

    def probe_regression(y, name):
        m = np.isfinite(y)
        Xm, ym = X[m], y[m]
        if len(ym) < 100:
            return {"probe": name, "n": int(len(ym)), "note": "too_few"}
        Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=args.test_size, random_state=args.seed)
        model = Ridge(alpha=10.0)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        return {
            "probe": name,
            "n": int(len(ym)),
            "mae": float(mean_absolute_error(yte, pred)),
            "r2": float(r2_score(yte, pred)),
        }

    def probe_classification(y, name):
        m = np.isfinite(y)
        Xm, ym = X[m], y[m].astype(int)
        if len(np.unique(ym)) < 2 or len(ym) < 100:
            return {"probe": name, "n": int(len(ym)), "note": "too_few_or_one_class"}
        Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=args.test_size, random_state=args.seed, stratify=ym)
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)
        return {
            "probe": name,
            "n": int(len(ym)),
            "auroc": float(roc_auc_score(yte, p)),
            "acc": float(accuracy_score(yte, yhat)),
        }

    results = []
    age = pd.to_numeric(df.get(args.age_col), errors="coerce").to_numpy(dtype=float)
    sex = pd.to_numeric(df.get(args.sex_col), errors="coerce").to_numpy(dtype=float)
    accm = pd.to_numeric(df.get(args.accmean_col), errors="coerce").to_numpy(dtype=float)

    results.append(probe_regression(age, "resid->age"))
    results.append(probe_classification(sex, "resid->sex"))
    results.append(probe_regression(accm, "resid->acc_mean"))

    # Also probe resid_norm directly (should still relate to health/atypicality)
    # But this is a 1D target, not from X; we'll just correlate with covars.
    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0, 1])

    norm_corrs = {
        "corr(resid_norm, age)": corr(resid_norm, age),
        "corr(resid_norm, sex)": corr(resid_norm, sex),
        "corr(resid_norm, acc_mean)": corr(resid_norm, accm),
        "n": int(np.isfinite(resid_norm).sum()),
    }

    out = pd.DataFrame(results)
    print(out.to_string(index=False))
    print("\nResidual norm correlations:")
    for k, v in norm_corrs.items():
        print(f"  {k}: {v}")

    out.to_csv(os.path.join(args.resid_dir, "residual_probe_results.csv"), index=False)
    pd.DataFrame([norm_corrs]).to_csv(os.path.join(args.resid_dir, "residual_norm_correlations.csv"), index=False)
    print(f"\n[OK] wrote probes to {args.resid_dir}")


if __name__ == "__main__":
    main()
