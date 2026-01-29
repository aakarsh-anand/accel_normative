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
    num_cols = [cfg.accmean_col]
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


def save_model(model: Pipeline, cfg: NormativeConfig, out_path: str) -> None:
    payload = {"model": model, "cfg": cfg}
    joblib.dump(payload, out_path)


def load_model(path: str) -> Tuple[Pipeline, NormativeConfig]:
    payload = joblib.load(path)
    return payload["model"], payload["cfg"]
