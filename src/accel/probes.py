# src/accel/probes.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True, help="Directory containing embeddings_week.npz and subject_ids.csv")
    ap.add_argument("--covars_csv", required=True)
    ap.add_argument("--id_col", default="Participant ID")
    ap.add_argument("--age_col", default="Age")
    ap.add_argument("--sex_col", default="Sex")
    ap.add_argument("--accmean_col", default="Overall acceleration average")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    emb_path = os.path.join(args.emb_dir, "embeddings_week.npz")
    ids_path = os.path.join(args.emb_dir, "subject_ids.csv")

    emb = np.load(emb_path)["emb"]  # [N,D]
    ids = pd.read_csv(ids_path)["ParticipantID"].astype(str)

    cov = pd.read_csv(args.covars_csv)
    cov[args.id_col] = cov[args.id_col].astype(str)

    df = pd.DataFrame({args.id_col: ids})
    df = df.merge(cov, on=args.id_col, how="left")

    # Keep rows with embeddings and labels
    X = emb
    results = {}

    def probe_regression(y, name):
        m = np.isfinite(y)
        Xm = X[m]
        ym = y[m]
        if len(ym) < 100:
            results[name] = {"note": "too_few_samples", "n": int(len(ym))}
            return
        Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=args.test_size, random_state=args.seed)
        model = Ridge(alpha=10.0)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        results[name] = {
            "n": int(len(ym)),
            "mae": float(mean_absolute_error(yte, pred)),
            "r2": float(r2_score(yte, pred)),
        }

    def probe_classification(y, name):
        m = np.isfinite(y)
        Xm = X[m]
        ym = y[m].astype(int)
        if len(np.unique(ym)) < 2 or len(ym) < 100:
            results[name] = {"note": "too_few_or_one_class", "n": int(len(ym))}
            return
        Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=args.test_size, random_state=args.seed, stratify=ym)
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)
        results[name] = {
            "n": int(len(ym)),
            "auroc": float(roc_auc_score(yte, p)),
            "acc": float(accuracy_score(yte, yhat)),
        }

    # Age probe
    age = pd.to_numeric(df.get(args.age_col), errors="coerce").to_numpy(dtype=float)
    probe_regression(age, "age")

    # Sex probe (binary)
    sex = pd.to_numeric(df.get(args.sex_col), errors="coerce").to_numpy(dtype=float)
    probe_classification(sex, "sex")

    # Mean acceleration probe
    accm = pd.to_numeric(df.get(args.accmean_col), errors="coerce").to_numpy(dtype=float)
    probe_regression(accm, "acc_mean")

    out = pd.DataFrame(results).T.reset_index().rename(columns={"index": "probe"})
    out_path = os.path.join(args.emb_dir, "probe_results.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
