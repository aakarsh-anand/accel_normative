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
    ap.add_argument("--wear_dates_csv", default=None, help="Optional CSV with wear_month_sin/cos columns")
    ap.add_argument("--outdir", required=True)
    
    # Column names
    ap.add_argument("--id_col", default="Participant ID")
    ap.add_argument("--age_col", default="Age")
    ap.add_argument("--sex_col", default="Sex")
    ap.add_argument("--accmean_col", default="Overall acceleration average")
    ap.add_argument("--bmi_col", default="BMI")
    ap.add_argument("--wear_col", default="Wear duration overall")
    ap.add_argument("--height_col", default="Standing Height")
    ap.add_argument("--townsend_col", default="Townsend Index")
    ap.add_argument("--smoking_col", default="Current smoking status")

    # Spline settings
    ap.add_argument("--alpha", type=float, default=None, help="Ridge alpha (ignored if --cv_alphas is set)")
    ap.add_argument("--cv_alphas", type=str, default=None, 
                    help="Comma-separated alphas for cross-validation, e.g. '0.1,1,10,100'")
    ap.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--n_knots", type=int, default=8)
    ap.add_argument("--degree", type=int, default=3)

    # Model type
    ap.add_argument("--per_dimension", action="store_true", 
                    help="Fit D separate models instead of one multi-output model")

    # Reference cohort
    ap.add_argument("--ref_ids_csv", default=None, help="Optional CSV with reference cohort IDs")
    ap.add_argument("--ref_id_col", default=None, help="ID column name in ref_ids_csv")
    
    # Geometry
    ap.add_argument("--geom_var", type=float, default=0.90, help="PCA variance for residual geometry")
    
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load embeddings
    emb = np.load(os.path.join(args.emb_dir, "embeddings_week.npz"))["emb"]  # [N,D]
    ids = pd.read_csv(os.path.join(args.emb_dir, "subject_ids.csv"))["ParticipantID"].astype(str)

    # Load covariates
    cov = pd.read_csv(args.covars_csv)
    cov[args.id_col] = cov[args.id_col].astype(str)

    df = pd.DataFrame({args.id_col: ids})
    df = df.merge(cov, on=args.id_col, how="left")

    # Merge wear dates if provided
    month_sin_col = None
    month_cos_col = None
    if args.wear_dates_csv is not None:
        wear_dates = pd.read_csv(args.wear_dates_csv)
        wear_dates[args.id_col] = wear_dates[args.id_col].astype(str)
        df = df.merge(wear_dates[[args.id_col, 'wear_month_sin', 'wear_month_cos']], 
                     on=args.id_col, how="left")
        month_sin_col = "wear_month_sin"
        month_cos_col = "wear_month_cos"
        print(f"[Season] Added seasonal covariates from {args.wear_dates_csv}")

    # Build reference mask if provided
    ref_mask = None
    if args.ref_ids_csv is not None:
        ref = pd.read_csv(args.ref_ids_csv)
        ref_id_col = args.ref_id_col or args.id_col
        ref_ids = set(ref[ref_id_col].astype(str).tolist())
        ref_mask = df[args.id_col].astype(str).isin(ref_ids).to_numpy()
        print(f"[Ref] Using {ref_mask.sum()} / {len(ref_mask)} subjects for normative fit")

    # Parse CV alphas
    cv_alphas = None
    if args.cv_alphas is not None:
        cv_alphas = [float(a) for a in args.cv_alphas.split(',')]
        print(f"[CV] Cross-validating alphas: {cv_alphas}")
        alpha = 10.0  # default, will be ignored
    else:
        alpha = args.alpha if args.alpha is not None else 10.0
        print(f"[Ridge] Using fixed alpha: {alpha}")

    # Build config
    cfg = NormativeConfig(
        id_col=args.id_col,
        age_col=args.age_col,
        sex_col=args.sex_col,
        accmean_col=args.accmean_col,
        bmi_col=args.bmi_col,
        wear_col=args.wear_col,
        height_col=args.height_col,
        townsend_col=args.townsend_col,
        smoking_col=args.smoking_col,
        month_sin_col=month_sin_col,
        month_cos_col=month_cos_col,
        n_knots=args.n_knots,
        degree=args.degree,
        alpha=alpha,
        cv_alphas=cv_alphas,
        cv_folds=args.cv_folds,
        per_dimension=args.per_dimension,
    )

    # Fit normative model
    print(f"\n[Fit] Starting normative model fitting...")
    if args.per_dimension:
        print(f"[Fit] Mode: Per-dimension (D={emb.shape[1]} separate models)")
    else:
        print(f"[Fit] Mode: Multi-output (single model predicting all {emb.shape[1]} dimensions)")
    
    model, metrics = fit_normative_model(emb=emb, covars=df, cfg=cfg, ref_mask=ref_mask)

    # Fit residual geometry
    from .normative import _get_covariate_columns
    covar_cols = _get_covariate_columns(cfg)
    Xdf = df[covar_cols].copy()
    
    if ref_mask is None:
        ref_mask_geom = np.ones(len(df), dtype=bool)
    else:
        ref_mask_geom = ref_mask

    print(f"\n[Geom] Fitting residual geometry on {ref_mask_geom.sum()} reference subjects...")
    
    # Predict on reference set
    if isinstance(model, list):
        # Per-dimension models
        D = len(model)
        N_ref = ref_mask_geom.sum()
        mu_fit = np.zeros((N_ref, D), dtype=np.float32)
        Xdf_ref = Xdf.loc[ref_mask_geom]
        for d, model_d in enumerate(model):
            mu_fit[:, d] = model_d.predict(Xdf_ref).astype(np.float32)
    else:
        # Multi-output model
        mu_fit = model.predict(Xdf.loc[ref_mask_geom])
    
    resid_fit = emb[ref_mask_geom] - mu_fit
    geometry = fit_residual_geometry(resid_fit=resid_fit, var_explained=args.geom_var)

    # Save outputs
    geom_path = os.path.join(args.outdir, "geometry.joblib")
    joblib.dump(geometry, geom_path)
    print(f"[OK] Saved geometry: {geom_path}")

    out_model = os.path.join(args.outdir, "normative_model.joblib")
    save_model(model, cfg, out_model)
    print(f"[OK] Saved model: {out_model}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.outdir, "normative_fit_metrics.csv"), index=False)
    print(f"[OK] Saved metrics: normative_fit_metrics.csv")
    
    print("\n" + "="*60)
    print("NORMATIVE MODEL FIT SUMMARY")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    main()