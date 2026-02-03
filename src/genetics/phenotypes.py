# src/genetics/phenotypes.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

def _infer_score_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer score column. Available: {list(df.columns)}")

def load_labels(labels_csv: str | Path, id_col: str) -> pd.DataFrame:
    lab = pd.read_csv(labels_csv)
    lab[id_col] = lab[id_col].astype(str)
    if "Label" not in lab.columns:
        raise ValueError(f"{labels_csv} missing 'Label' column")
    return lab[[id_col, "Label"]].copy()

def load_axis_scores(scores_csv: str | Path, id_col: str, score_col_candidates: list[str]) -> pd.DataFrame:
    sc = pd.read_csv(scores_csv)
    sc[id_col] = sc[id_col].astype(str)
    score_col = _infer_score_col(sc, score_col_candidates)
    sc = sc[[id_col, score_col]].rename(columns={score_col: "axis_score"})
    return sc

def standardize(series: pd.Series) -> pd.Series:
    x = series.astype(float)
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

def make_binary_pheno(ids: pd.Series, labels: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = pd.DataFrame({id_col: ids.astype(str)})
    df = df.merge(labels, on=id_col, how="left")
    df["Label"] = df["Label"].fillna(0).astype(int)
    # plink expects 1/2 for case/control sometimes; we keep 0/1 and let wrapper map if needed
    df["PHENO"] = df["Label"]
    return df[[id_col, "PHENO"]]

def make_continuous_pheno(ids: pd.Series, scores: pd.DataFrame, id_col: str, zscore: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({id_col: ids.astype(str)}).merge(scores, on=id_col, how="left")
    if df["axis_score"].isna().all():
        raise ValueError("All axis_score are NaN after merge; check ids alignment")
    ph = df["axis_score"]
    if zscore:
        ph = standardize(ph)
    df["PHENO"] = ph
    return df[[id_col, "PHENO"]]

def make_control_replaced_pheno(
    ids: pd.Series,
    labels: pd.DataFrame,
    scores: pd.DataFrame,
    id_col: str,
    zscore: bool = True,
) -> pd.DataFrame:
    """
    Variant requested: "only replaces controls with continuous values".

    Implemented as:
      - if Label==1 (case): PHENO = 1
      - if Label==0 (control): PHENO = z-scored axis_score (or raw if zscore=False)

    This preserves case indicator while enriching controls with continuous variation.
    """
    df = pd.DataFrame({id_col: ids.astype(str)})
    df = df.merge(labels, on=id_col, how="left")
    df["Label"] = df["Label"].fillna(0).astype(int)
    df = df.merge(scores, on=id_col, how="left")

    control_score = df["axis_score"].astype(float)
    if zscore:
        control_score = standardize(control_score)

    ph = control_score.copy()
    ph[df["Label"] == 1] = 1.0
    df["PHENO"] = ph
    return df[[id_col, "PHENO"]]

def to_plink_pheno(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    out["FID"] = out[id_col].astype(str)
    out["IID"] = out[id_col].astype(str)
    out = out[["FID", "IID", "PHENO"]]
    return out
