# src/accel/residual_probes.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, accuracy_score


def _to_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _to_class_codes(series: pd.Series) -> tuple[np.ndarray, dict[int, object]]:
    """
    Convert a pandas Series to integer class codes.
    Missing values become -1. Returns (codes, mapping[int->label]).
    """
    s = series.copy()
    # Normalize missing tokens
    if s.dtype == object:
        s = s.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "nan": np.nan})

    # If already numeric, keep as-is (but coerce)
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x)
        codes = np.full_like(x, fill_value=-1, dtype=int)
        # Map unique finite values to 0..K-1
        uniq = np.unique(x[m])
        mapping = {i: float(v) for i, v in enumerate(uniq.tolist())}
        inv = {float(v): i for i, v in enumerate(uniq.tolist())}
        codes[m] = np.vectorize(inv.get)(x[m])
        return codes, mapping

    # Otherwise treat as categorical
    cat = s.astype("category")
    codes = cat.cat.codes.to_numpy(dtype=int)  # -1 for NaN
    mapping = {int(i): lab for i, lab in enumerate(cat.cat.categories.tolist())}
    return codes, mapping


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _eta_squared(y: np.ndarray, g: np.ndarray) -> float:
    """
    Effect size for categorical g on numeric y (one-way ANOVA eta^2).
    y: numeric, g: int codes with -1 missing.
    """
    m = np.isfinite(y) & (g >= 0)
    if m.sum() < 100:
        return float("nan")
    y = y[m]
    g = g[m]
    if np.unique(g).size < 2:
        return float("nan")
    grand = y.mean()
    ss_total = np.sum((y - grand) ** 2)
    ss_between = 0.0
    for k in np.unique(g):
        yk = y[g == k]
        ss_between += yk.size * (yk.mean() - grand) ** 2
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resid_dir", required=True, help="Directory with normative_residuals.npz and residual_summary.csv")

    # Columns (mirror normative pipeline defaults)
    ap.add_argument("--id_col", default="Participant ID")
    ap.add_argument("--age_col", default="Age")
    ap.add_argument("--sex_col", default="Sex")
    ap.add_argument("--accmean_col", default="Overall acceleration average")
    ap.add_argument("--bmi_col", default="BMI")
    ap.add_argument("--wear_col", default="Wear duration overall")
    ap.add_argument("--height_col", default="Standing Height")
    ap.add_argument("--townsend_col", default="Townsend Index")
    ap.add_argument("--smoking_col", default="Current smoking status")

    # Optional seasonal adjustment cols (if present in residual_summary.csv)
    ap.add_argument("--month_sin_col", default="wear_month_sin")
    ap.add_argument("--month_cos_col", default="wear_month_cos")

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    npz_path = os.path.join(args.resid_dir, "normative_residuals.npz")
    csv_path = os.path.join(args.resid_dir, "residual_summary.csv")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    z = np.load(npz_path)
    resid = z["resid"]  # [N, D]
    resid_norm = z["resid_norm"]  # [N]
    df = pd.read_csv(csv_path)

    X = resid

    def probe_regression(y: np.ndarray, name: str):
        m = np.isfinite(y)
        Xm, ym = X[m], y[m]
        if ym.size < 100:
            return {"probe": name, "type": "reg", "n": int(ym.size), "note": "too_few"}
        Xtr, Xte, ytr, yte = train_test_split(
            Xm, ym, test_size=args.test_size, random_state=args.seed
        )
        model = Ridge(alpha=10.0)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        return {
            "probe": name,
            "type": "reg",
            "n": int(ym.size),
            "mae": float(mean_absolute_error(yte, pred)),
            "r2": float(r2_score(yte, pred)),
        }

    def probe_classification(y_codes: np.ndarray, name: str):
        m = (y_codes >= 0) & np.isfinite(y_codes.astype(float))
        Xm, ym = X[m], y_codes[m].astype(int)
        n = int(ym.size)
        k = int(np.unique(ym).size) if n > 0 else 0
        if k < 2 or n < 100:
            return {"probe": name, "type": "clf", "n": n, "k": k, "note": "too_few_or_one_class"}

        # Stratify only if every class has at least 2 samples
        stratify = ym if np.all(np.bincount(ym) >= 2) else None

        Xtr, Xte, ytr, yte = train_test_split(
            Xm, ym, test_size=args.test_size, random_state=args.seed, stratify=stratify
        )

        model = LogisticRegression(max_iter=4000, solver="lbfgs")
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)  # [n_test, k]

        if k == 2:
            p = proba[:, 1]
            yhat = (p >= 0.5).astype(int)
            auroc = float(roc_auc_score(yte, p))
        else:
            yhat = np.argmax(proba, axis=1)
            # For multiclass roc_auc_score expects (n_samples, n_classes)
            auroc = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))

        return {
            "probe": name,
            "type": "clf",
            "n": n,
            "k": k,
            "auroc": auroc,
            "acc": float(accuracy_score(yte, yhat)),
        }

    # --- Build list of covariates to probe ---
    covar_specs: list[tuple[str, str]] = [
        ("reg", args.age_col),
        ("clf", args.sex_col),
        ("reg", args.accmean_col),
        ("reg", args.bmi_col),
        ("reg", args.wear_col),
        ("reg", args.height_col),
        ("reg", args.townsend_col),
        ("clf", args.smoking_col),
    ]

    # Optional seasonal covars if present
    if args.month_sin_col in df.columns:
        covar_specs.append(("reg", args.month_sin_col))
    if args.month_cos_col in df.columns:
        covar_specs.append(("reg", args.month_cos_col))

    results = []
    norm_assoc = {"n": int(np.isfinite(resid_norm).sum())}

    for kind, col in covar_specs:
        if col not in df.columns:
            results.append({"probe": f"resid->{col}", "type": kind, "n": 0, "note": "missing_column"})
            if kind == "reg":
                norm_assoc[f"corr(resid_norm, {col})"] = float("nan")
            else:
                norm_assoc[f"eta2(resid_norm ~ {col})"] = float("nan")
            continue

        s = df[col]

        if kind == "reg":
            y = _to_numeric(s)
            results.append(probe_regression(y, f"resid->{col}"))
            norm_assoc[f"corr(resid_norm, {col})"] = _corr(resid_norm, y)
        else:
            codes, mapping = _to_class_codes(s)
            row = probe_classification(codes, f"resid->{col}")
            # attach mapping info (compact) for interpretability
            if "note" not in row:
                row["class_map"] = {int(k): str(v) for k, v in mapping.items()}
            results.append(row)
            norm_assoc[f"eta2(resid_norm ~ {col})"] = _eta_squared(resid_norm.astype(float), codes)

    out = pd.DataFrame(results)
    print(out.to_string(index=False))

    print("\nResidual norm associations:")
    for k, v in norm_assoc.items():
        print(f"  {k}: {v}")

    out.to_csv(os.path.join(args.resid_dir, "residual_probe_results.csv"), index=False)
    pd.DataFrame([norm_assoc]).to_csv(os.path.join(args.resid_dir, "residual_norm_associations.csv"), index=False)
    print(f"\n[OK] wrote probes to {args.resid_dir}")


if __name__ == "__main__":
    main()
