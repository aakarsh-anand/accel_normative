# src/accel/labels.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class PDLabelConfig:
    id_col_icd: str = "participant.eid"
    accel_start_col: str = "participant.p90003"  # start of accel wear
    # list of ICD date columns that indicate PD diagnosis (datetimes)
    pd_date_cols: List[str] = None


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def compute_pd_tte_and_status(
    icd_df: pd.DataFrame,
    cfg: PDLabelConfig,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - Participant ID (string)
      - accel_start_date (datetime)
      - pd_date (datetime) = earliest available PD diagnosis date across columns
      - TTE_years (float) = (pd_date - accel_start_date)/365.25
      - Status = Control / Prodromal / Diagnosed
      - Label = 1 if PD ever (has pd_date) else 0
    """
    if cfg.pd_date_cols is None or len(cfg.pd_date_cols) == 0:
        raise ValueError("PDLabelConfig.pd_date_cols must be a non-empty list of ICD date columns for PD.")

    df = icd_df.copy()
    df[cfg.id_col_icd] = df[cfg.id_col_icd].astype(str)

    accel_start = _to_datetime(df[cfg.accel_start_col])
    pd_dates = []
    for c in cfg.pd_date_cols:
        if c not in df.columns:
            raise ValueError(f"PD ICD date column not found: {c}")
        pd_dates.append(_to_datetime(df[c]))

    # earliest PD date across PD columns
    pd_min = pd.concat(pd_dates, axis=1).min(axis=1)

    tte_years = (pd_min - accel_start).dt.total_seconds() / (365.25 * 24 * 3600)

    status = np.array(["Control"] * len(df), dtype=object)
    status[(tte_years > 0).fillna(False).to_numpy()] = "Prodromal"
    status[(tte_years <= 0).fillna(False).to_numpy()] = "Diagnosed"

    out = pd.DataFrame({
        "Participant ID": df[cfg.id_col_icd].astype(str),
        "accel_start_date": accel_start,
        "pd_date": pd_min,
        "TTE_years": pd.to_numeric(tte_years, errors="coerce"),
        "Status": pd.Categorical(status, categories=["Control","Prodromal","Diagnosed"], ordered=True),
    })
    out["Label"] = (out["pd_date"].notna()).astype(int)
    return out
