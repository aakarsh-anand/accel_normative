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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib


@dataclass
class NormativeConfig:
    id_col: str = "Participant ID"
    age_col: str = "Age"
    sex_col: str = "Sex"
    accmean_col: str = "Overall acceleration average"
    bmi_col: str = "BMI"
    wear_col: str = "Wear duration overall"
    month_col: Optional[str] = None

    # Spline settings
    n_knots: int = 8
    degree: int = 3

    # Model
    alpha: float = 10.0  # ridge strength
    standardize_y: bool = False  # usually not needed for ridge


def _build_design_pipeline(cfg: NormativeConfig) -> ColumnTransformer:
    """
    Build a ColumnTransformer for:
      - age spline features
      - sex one-hot (or passthrough if already 0/1)
      - acc_mean numeric
    """
    num_cols = [cfg.accmean_col, cfg.bmi_col, cfg.wear_col]
    # sex treated as categorical so it doesn't assume linearity
    cat_cols = [cfg.sex_col]
    age_cols = [cfg.age_col]

    age_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=cfg.n_knots, degree=cfg.degree, include_bias=False)),
        ("scaler", StandardScaler(with_mean=False)),  # sparse-friendly; spline output can be dense but fine
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("age_spline", age_pipe, age_cols),
            ("num", num_pipe, num_cols),
            ("sex", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct


def fit_normative_model(
    emb: np.ndarray,                 # [N, D]
    covars: pd.DataFrame,            # must align with ids used for emb
    cfg: NormativeConfig,
    ref_mask: Optional[np.ndarray] = None,  # boolean mask length N for "reference/healthy" fitting
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Fit multi-output ridge to predict embedding from (age spline + sex + acc_mean).
    Returns:
      model: Pipeline(design -> Ridge)
      metrics: dict with overall R2 (mean across dims), and median R2
    """
    if emb.ndim != 2:
        raise ValueError(f"emb must be [N,D], got {emb.shape}")

    N, D = emb.shape
    if len(covars) != N:
        raise ValueError(f"covars rows ({len(covars)}) must match emb N ({N})")

    Xdf = covars[[cfg.age_col, cfg.sex_col, cfg.accmean_col]].copy()

    if ref_mask is None:
        ref_mask = np.ones(N, dtype=bool)

    X_fit = Xdf.loc[ref_mask]
    Y_fit = emb[ref_mask]

    design = _build_design_pipeline(cfg)
    ridge = Ridge(alpha=cfg.alpha, fit_intercept=True, random_state=0)

    model = Pipeline([
        ("design", design),
        ("ridge", ridge),
    ])

    model.fit(X_fit, Y_fit)

    # diagnostics on the fit set (not a generalization claim; just sanity)
    Y_pred = model.predict(X_fit)
    r2_dims = r2_score(Y_fit, Y_pred, multioutput="raw_values")  # [D]
    metrics = {
        "fit_r2_mean": float(np.mean(r2_dims)),
        "fit_r2_median": float(np.median(r2_dims)),
        "fit_r2_p10": float(np.percentile(r2_dims, 10)),
        "fit_r2_p90": float(np.percentile(r2_dims, 90)),
        "n_fit": int(ref_mask.sum()),
        "n_total": int(N),
        "D": int(D),
    }
    return model, metrics


def predict_mu_and_residuals(
    model: Pipeline,
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
    Xdf = covars[[cfg.age_col, cfg.sex_col, cfg.accmean_col]].copy()
    mu = model.predict(Xdf).astype(np.float32)
    resid = (emb.astype(np.float32) - mu).astype(np.float32)
    resid_norm = np.sqrt(np.sum(resid * resid, axis=1)).astype(np.float32)
    return mu, resid, resid_norm

# ---- Residual geometry (Phase 1 upgrade) ----
# This is intentionally kept separate from the mean model fit: the mean model
# predicts E[emb | covars], while geometry summarizes the residual distribution
# on a chosen reference cohort (e.g., controls).

def fit_residual_geometry(
    resid_fit: np.ndarray,
    var_explained: float = 0.90,
) -> Dict[str, object]:
    """Fit a low-rank Gaussian geometry to residuals.

    Args:
        resid_fit: [N_ref, D] residual vectors on the *reference* cohort.
        var_explained: PCA variance to retain (0<var<=1).

    Returns:
        geometry dict containing:
          - 'pca': sklearn PCA object
          - 'cov': covariance matrix in PCA space [K,K]
          - 'var_explained': float
          - 'eps': float used for numerical stability in whitening
    """
    if resid_fit.ndim != 2:
        raise ValueError(f"resid_fit must be [N,D], got {resid_fit.shape}")

    from sklearn.decomposition import PCA

    pca = PCA(n_components=var_explained, svd_solver="full", random_state=0)
    R = pca.fit_transform(resid_fit)

    cov = np.cov(R, rowvar=False)
    if cov.ndim == 0:
        # K==1 edge case
        cov = np.array([[float(cov)]], dtype=float)

    return {
        "pca": pca,
        "cov": cov.astype(float),
        "var_explained": float(var_explained),
        "eps": 1e-6,
    }


def whiten_residuals(resid: np.ndarray, geometry: Dict[str, object]) -> np.ndarray:
    """Whiten residuals using reference-set PCA+covariance.

    Returns:
        resid_w: [N,K] whitened residuals in PCA space.
    """
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



def save_model(model: Pipeline, cfg: NormativeConfig, out_path: str) -> None:
    payload = {"model": model, "cfg": cfg}
    joblib.dump(payload, out_path)


def load_model(path: str) -> Tuple[Pipeline, NormativeConfig]:
    payload = joblib.load(path)
    return payload["model"], payload["cfg"]