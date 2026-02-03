# src/accel/normative.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import joblib


@dataclass
class NormativeConfig:
    id_col: str = "Participant ID"
    age_col: str = "Age"
    sex_col: str = "Sex"
    accmean_col: str = "Overall acceleration average"
    bmi_col: str = "BMI"
    wear_col: str = "Wear duration overall"
    height_col: str = "Standing Height"
    townsend_col: str = "Townsend Index"
    smoking_col: str = "Current smoking status"
    
    # Optional seasonal adjustment
    month_sin_col: Optional[str] = "wear_month_sin"
    month_cos_col: Optional[str] = "wear_month_cos"

    # Spline settings
    n_knots: int = 8
    degree: int = 3

    # Model
    alpha: float = 10.0  # ridge strength
    standardize_y: bool = False
    
    # Cross-validation
    cv_alphas: Optional[List[float]] = None  # If set, use RidgeCV with these alphas
    cv_folds: int = 5
    
    # Per-dimension modeling
    per_dimension: bool = False  # If True, fit D separate models instead of multi-output


def _build_design_pipeline(cfg: NormativeConfig) -> ColumnTransformer:
    """
    Build a ColumnTransformer for:
      - age spline features
      - categorical features (sex, smoking)
      - numeric features (acc_mean, bmi, wear_time, height, townsend)
      - optional: cyclic month features
    """
    # Numeric covariates
    num_cols = [
        cfg.accmean_col,
        cfg.bmi_col,
        cfg.wear_col,
        cfg.height_col,
        cfg.townsend_col,
    ]

    # Categorical covariates (note: smoking can be multi-class)
    cat_cols = [cfg.sex_col, cfg.smoking_col]
    age_cols = [cfg.age_col]
    
    transformers = []
    
    # Age spline
    age_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=cfg.n_knots, degree=cfg.degree, include_bias=False)),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    transformers.append(("age_spline", age_pipe, age_cols))
    
    # Numeric features
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers.append(("num", num_pipe, num_cols))
    
    # Categorical features
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers.append(("cat", cat_pipe, cat_cols))
    
    # Optional: cyclic month features
    if cfg.month_sin_col is not None and cfg.month_cos_col is not None:
        month_cols = [cfg.month_sin_col, cfg.month_cos_col]
        month_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("month", month_pipe, month_cols))
    
    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct


def _get_covariate_columns(cfg: NormativeConfig) -> List[str]:
    """Get list of all covariate columns needed."""
    cols = [cfg.age_col, cfg.sex_col, cfg.accmean_col, cfg.bmi_col, cfg.wear_col, cfg.height_col, cfg.townsend_col, cfg.smoking_col]
    if cfg.month_sin_col is not None:
        cols.append(cfg.month_sin_col)
    if cfg.month_cos_col is not None:
        cols.append(cfg.month_cos_col)
    return cols


def fit_normative_model(
    emb: np.ndarray,                 # [N, D]
    covars: pd.DataFrame,            # must align with ids used for emb
    cfg: NormativeConfig,
    ref_mask: Optional[np.ndarray] = None,  # boolean mask length N
) -> Tuple[Pipeline | List[Pipeline], Dict[str, float]]:
    """
    Fit ridge regression to predict embedding from covariates.
    
    Returns:
      model: Pipeline (multi-output) OR List[Pipeline] (per-dimension models)
      metrics: dict with R2 statistics
    """
    if emb.ndim != 2:
        raise ValueError(f"emb must be [N,D], got {emb.shape}")

    N, D = emb.shape
    if len(covars) != N:
        raise ValueError(f"covars rows ({len(covars)}) must match emb N ({N})")

    covar_cols = _get_covariate_columns(cfg)
    Xdf = covars[covar_cols].copy()

    if ref_mask is None:
        ref_mask = np.ones(N, dtype=bool)

    X_fit = Xdf.loc[ref_mask]
    Y_fit = emb[ref_mask]

    design = _build_design_pipeline(cfg)
    
    # Determine whether to use cross-validation
    if cfg.cv_alphas is not None:
        print(f"[CV] Using RidgeCV with alphas={cfg.cv_alphas}, cv={cfg.cv_folds}")
        ridge = RidgeCV(
            alphas=cfg.cv_alphas,
            cv=cfg.cv_folds,
            fit_intercept=True,
            scoring='r2',
        )
    else:
        ridge = Ridge(alpha=cfg.alpha, fit_intercept=True, random_state=0)
    
    if not cfg.per_dimension:
        # Multi-output model (original approach)
        model = Pipeline([
            ("design", design),
            ("ridge", ridge),
        ])
        model.fit(X_fit, Y_fit)
        
        # Diagnostics
        Y_pred = model.predict(X_fit)
        r2_dims = r2_score(Y_fit, Y_pred, multioutput="raw_values")
        
        # If using CV, report selected alpha
        if cfg.cv_alphas is not None:
            selected_alpha = model.named_steps['ridge'].alpha_
            print(f"[CV] Selected alpha: {selected_alpha}")
        
        metrics = {
            "fit_r2_mean": float(np.mean(r2_dims)),
            "fit_r2_median": float(np.median(r2_dims)),
            "fit_r2_p10": float(np.percentile(r2_dims, 10)),
            "fit_r2_p90": float(np.percentile(r2_dims, 90)),
            "n_fit": int(ref_mask.sum()),
            "n_total": int(N),
            "D": int(D),
            "per_dimension": False,
        }
        if cfg.cv_alphas is not None:
            metrics["selected_alpha"] = float(selected_alpha)
        
        return model, metrics
    
    else:
        # Per-dimension models
        print(f"[PerDim] Fitting {D} separate models...")
        models = []
        r2_dims = np.zeros(D)
        selected_alphas = np.zeros(D) if cfg.cv_alphas is not None else None
        
        for d in range(D):
            if d % 10 == 0:
                print(f"  Fitting dimension {d+1}/{D}...")
            
            # Create fresh pipeline for this dimension
            design_d = _build_design_pipeline(cfg)
            if cfg.cv_alphas is not None:
                ridge_d = RidgeCV(
                    alphas=cfg.cv_alphas,
                    cv=cfg.cv_folds,
                    fit_intercept=True,
                    scoring='r2',
                )
            else:
                ridge_d = Ridge(alpha=cfg.alpha, fit_intercept=True, random_state=0)
            
            model_d = Pipeline([
                ("design", design_d),
                ("ridge", ridge_d),
            ])
            
            model_d.fit(X_fit, Y_fit[:, d])
            models.append(model_d)
            
            # Compute R2 for this dimension
            y_pred_d = model_d.predict(X_fit)
            r2_dims[d] = r2_score(Y_fit[:, d], y_pred_d)
            
            if cfg.cv_alphas is not None:
                selected_alphas[d] = model_d.named_steps['ridge'].alpha_
        
        print(f"[PerDim] Done. Mean R2: {np.mean(r2_dims):.4f}")
        
        metrics = {
            "fit_r2_mean": float(np.mean(r2_dims)),
            "fit_r2_median": float(np.median(r2_dims)),
            "fit_r2_p10": float(np.percentile(r2_dims, 10)),
            "fit_r2_p90": float(np.percentile(r2_dims, 90)),
            "n_fit": int(ref_mask.sum()),
            "n_total": int(N),
            "D": int(D),
            "per_dimension": True,
        }
        
        if cfg.cv_alphas is not None:
            metrics["selected_alpha_mean"] = float(np.mean(selected_alphas))
            metrics["selected_alpha_median"] = float(np.median(selected_alphas))
            metrics["selected_alpha_std"] = float(np.std(selected_alphas))
        
        return models, metrics


def predict_mu_and_residuals(
    model: Pipeline | List[Pipeline],
    emb: np.ndarray,            # [N,D]
    covars: pd.DataFrame,
    cfg: NormativeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      mu: [N,D]
      resid: [N,D] = emb - mu
      resid_norm: [N]
    """
    covar_cols = _get_covariate_columns(cfg)
    Xdf = covars[covar_cols].copy()
    
    if isinstance(model, Pipeline):
        # Multi-output model
        mu = model.predict(Xdf).astype(np.float32)
    else:
        # Per-dimension models
        D = len(model)
        N = len(Xdf)
        mu = np.zeros((N, D), dtype=np.float32)
        for d, model_d in enumerate(model):
            mu[:, d] = model_d.predict(Xdf).astype(np.float32)
    
    resid = (emb.astype(np.float32) - mu).astype(np.float32)
    resid_norm = np.sqrt(np.sum(resid * resid, axis=1)).astype(np.float32)
    return mu, resid, resid_norm


def fit_residual_geometry(
    resid_fit: np.ndarray,
    var_explained: float = 0.90,
) -> Dict[str, object]:
    """Fit a low-rank Gaussian geometry to residuals."""
    if resid_fit.ndim != 2:
        raise ValueError(f"resid_fit must be [N,D], got {resid_fit.shape}")

    from sklearn.decomposition import PCA

    pca = PCA(n_components=var_explained, svd_solver="full", random_state=0)
    R = pca.fit_transform(resid_fit)

    cov = np.cov(R, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)

    return {
        "pca": pca,
        "cov": cov.astype(float),
        "var_explained": float(var_explained),
        "eps": 1e-6,
    }


def whiten_residuals(resid: np.ndarray, geometry: Dict[str, object]) -> np.ndarray:
    """Whiten residuals using reference-set PCA+covariance."""
    pca = geometry["pca"]
    cov = np.asarray(geometry["cov"], dtype=float)
    eps = float(geometry.get("eps", 1e-6))

    R = pca.transform(resid)  # [N,K]
    eigvals, eigvecs = np.linalg.eigh(cov)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    return (R @ W).astype(np.float32)


def mahalanobis_norm(resid_w: np.ndarray) -> np.ndarray:
    """Mahalanobis-like norm = L2 norm in whitened PCA space."""
    return np.sqrt(np.sum(resid_w * resid_w, axis=1)).astype(np.float32)


def save_model(model: Pipeline | List[Pipeline], cfg: NormativeConfig, out_path: str) -> None:
    payload = {"model": model, "cfg": cfg}
    joblib.dump(payload, out_path)


def load_model(path: str) -> Tuple[Pipeline | List[Pipeline], NormativeConfig]:
    payload = joblib.load(path)
    return payload["model"], payload["cfg"]