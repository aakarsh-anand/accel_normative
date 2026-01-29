from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Constants derived from your population summary
GLOBAL_MEDIAN = 5
GLOBAL_IQR = 30

def _global_robust_scale(x: np.ndarray, clip: float) -> np.ndarray:
    """Scale by global population stats to preserve absolute intensity."""
    z = (x - GLOBAL_MEDIAN) / GLOBAL_IQR
    return np.clip(z, -clip, clip)

def _safe_read_csv(path: str) -> pd.DataFrame:
    # Pandas can be slow; keep it simple for now. You can swap to pyarrow later.
    return pd.read_csv(path)


def _impute_nan_1d(x: np.ndarray) -> np.ndarray:
    """Impute NaNs with linear interpolation; fallback to ffill/bfill."""
    if not np.isnan(x).any():
        return x
    n = len(x)
    idx = np.arange(n)
    good = ~np.isnan(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    # linear interp on good points
    x_interp = np.interp(idx, idx[good], x[good]).astype(x.dtype, copy=False)
    return x_interp


def _robust_scale(x: np.ndarray, mask_valid: np.ndarray, clip: float) -> np.ndarray:
    """Robustly scale by median and IQR on valid points."""
    v = x[mask_valid]
    if v.size < 10:
        return np.clip(x, -clip, clip)
    med = np.median(v)
    q25 = np.percentile(v, 25)
    q75 = np.percentile(v, 75)
    iqr = max(q75 - q25, 1e-6)
    z = (x - med) / iqr
    return np.clip(z, -clip, clip)


@dataclass
class WindowConfig:
    window_hours: int = 12
    sample_rate_seconds: int = 5
    windows_per_subject: int = 2
    min_fraction_nonmissing: float = 0.7
    robust_scale: bool = True
    clip_value: float = 8.0


class UKBAccelWindowDataset(Dataset):
    """
    Each item returns multiple sampled windows for one subject:
      x: [K, C=2, L]  (accel, missing/imputed mask)
      mask: [K, 1, L] (MAE mask: 1 = observed, 0 = masked out for encoder input)
    """
    def __init__(
        self,
        accel_dir: str,
        covars_csv: str,
        id_col: str,
        max_subjects: Optional[int] = None,
        window_cfg: WindowConfig = WindowConfig(),
        seed: int = 7,
    ):
        self.accel_dir = accel_dir
        self.id_col = id_col
        self.window_cfg = window_cfg

        cov = pd.read_csv(covars_csv)
        if id_col not in cov.columns:
            raise ValueError(f"covars_csv missing id_col={id_col}")
        ids = cov[id_col].astype(str).tolist()

        # Only keep IDs with a matching file
        files = []
        keep_ids = []
        for sid in ids:
            f = os.path.join(accel_dir, f"{sid}_90004_0_0.csv")
            if os.path.exists(f):
                keep_ids.append(sid)
                files.append(f)

        if max_subjects is not None:
            keep_ids = keep_ids[:max_subjects]
            files = files[:max_subjects]

        self.subject_ids = keep_ids
        self.files = files
        self.rng = random.Random(seed)

        self.L = int((window_cfg.window_hours * 3600) / window_cfg.sample_rate_seconds)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _load_subject(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        df = _safe_read_csv(path)

        x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float32)
        imp = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(1).to_numpy(dtype=np.float32)
        # NaNs in x are additional missing
        nan_mask = np.isnan(x)
        x = _impute_nan_1d(x)
        m = ((imp > 0.5) | nan_mask).astype(np.float32)  # 1 means imputed/missing-like
        return x, m

    def _sample_window(self, x: np.ndarray, m: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        n = len(x)
        if n < self.L:
            return None
        # attempt a few times to find a window with enough non-missing
        for _ in range(20):
            start = self.rng.randint(0, n - self.L)
            xw = x[start:start+self.L].copy()
            mw = m[start:start+self.L].copy()
            # valid points: not missing/imputed
            valid = (mw < 0.5)
            if valid.mean() < self.window_cfg.min_fraction_nonmissing:
                continue

            if self.window_cfg.robust_scale:
                xw = _global_robust_scale(xw, self.window_cfg.clip_value)
            else:
                xw = np.clip(xw, -self.window_cfg.clip_value, self.window_cfg.clip_value)

            # Channelize: accel + missingness indicator
            # We keep missingness as-is; model can learn to ignore it.
            X = np.stack([xw, mw], axis=0)  # [2, L]
            return X, valid.astype(np.float32)  # valid used for recon weighting if desired
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        sid = self.subject_ids[idx]
        x, m = self._load_subject(path)

        K = self.window_cfg.windows_per_subject
        windows = []
        valids = []
        for _ in range(K):
            out = self._sample_window(x, m)
            if out is None:
                # fallback: just take first L (or pad) if we must
                if len(x) >= self.L:
                    xw = x[:self.L].copy()
                    mw = m[:self.L].copy()
                    valid = (mw < 0.5)
                    if self.window_cfg.robust_scale:
                        xw = _robust_scale(xw, valid, self.window_cfg.clip_value)
                    X = np.stack([xw, mw], axis=0)
                    windows.append(X)
                    valids.append(valid.astype(np.float32))
            else:
                X, valid = out
                windows.append(X)
                valids.append(valid)

        X = np.stack(windows, axis=0).astype(np.float32)      # [K, 2, L]
        V = np.stack(valids, axis=0).astype(np.float32)       # [K, L]

        return {
            "subject_id": sid,
            "x": torch.from_numpy(X),
            "valid": torch.from_numpy(V),
        }
