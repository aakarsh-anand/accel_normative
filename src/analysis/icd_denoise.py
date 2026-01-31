# src/analysis/icd_denoise.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def knn_enrichment(
    X: np.ndarray,
    ids: np.ndarray,
    labels: pd.DataFrame,
    id_col: str = "Participant ID",
    label_col: str = "Label",
    k: int = 50,
) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors

    id_to_row = {str(i): r for r, i in enumerate(ids.astype(str))}
    lab = labels.copy()
    lab["row"] = lab[id_col].astype(str).map(id_to_row)
    lab = lab.dropna(subset=["row"])
    lab["row"] = lab["row"].astype(int)

    y = np.zeros(len(ids), dtype=int)
    y[lab["row"].values] = lab[label_col].astype(int).values

    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
    neigh = nbrs.kneighbors(X, return_distance=False)[:, 1:]  # exclude self

    frac_case = y[neigh].mean(axis=1)

    out = pd.DataFrame({
        id_col: ids.astype(str),
        "neighbor_case_frac": frac_case,
        "label": y,
    })
    return out

def candidate_relabels(df: pd.DataFrame, high_q: float = 0.99, low_q: float = 0.01) -> pd.DataFrame:
    hi = df["neighbor_case_frac"].quantile(high_q)
    lo = df["neighbor_case_frac"].quantile(low_q)

    out = df.copy()
    out["candidate_fp"] = (out["label"] == 0) & (out["neighbor_case_frac"] >= hi)
    out["candidate_fn"] = (out["label"] == 1) & (out["neighbor_case_frac"] <= lo)
    out["high_thr"] = hi
    out["low_thr"] = lo
    return out
