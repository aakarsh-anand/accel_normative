# src/analysis/deepen.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .viz import savefig

@dataclass
class DeepenConfig:
    id_col: str = "Participant ID"
    score_col: str = "axis_score"     # standardize this in runner
    tte_col: str = "TTE_years"
    status_col: str = "Status"
    leadtime_min: float = -10.0
    leadtime_max: float = 0.0
    leadtime_bins: int = 40
    high_score_quantile: float = 0.95

def load_residual_npz(resid_dir: str | Path) -> dict:
    npz_path = Path(resid_dir) / "normative_residuals.npz"
    return dict(np.load(npz_path))

def load_residual_summary(resid_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(resid_dir) / "residual_summary.csv")

def load_labels(labels_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(labels_csv)

def load_axis_scores(axis_scores_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(axis_scores_csv)

def merge_subject_table(
    resid_summary: pd.DataFrame,
    labels: pd.DataFrame,
    axis_scores: pd.DataFrame,
    cfg: DeepenConfig,
) -> pd.DataFrame:
    # Labels CSV is authoritative for Label/Status/TTE_years
    df = resid_summary.merge(labels, on=cfg.id_col, how="inner")

    # Axis scores CSV is authoritative only for score column (avoid duplicated Label/Status/TTE_years)
    keep_cols = [cfg.id_col, cfg.score_col]
    missing = [c for c in keep_cols if c not in axis_scores.columns]
    if missing:
        raise ValueError(f"axis_scores is missing columns {missing}. Available: {list(axis_scores.columns)}")
    axis_scores_small = axis_scores[keep_cols].drop_duplicates(subset=[cfg.id_col])

    df = df.merge(axis_scores_small, on=cfg.id_col, how="inner")
    return df

def plot_offramp_umap(
    X: np.ndarray,
    df: pd.DataFrame,
    out_png: str | Path,
    cfg: DeepenConfig,
    color_by: str = "Status",
):
    # UMAP import here so repo doesn't hard-require it unless used.
    import umap

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.15,
        n_components=2,
        random_state=0,
        metric="euclidean",
    )
    emb2 = reducer.fit_transform(X)

    plt.figure(figsize=(6.5, 5.5))
    if color_by in df.columns:
        vals = df[color_by].astype(str).values
        # simple categorical scatter
        for v in pd.unique(vals):
            m = vals == v
            plt.scatter(emb2[m, 0], emb2[m, 1], s=4, alpha=0.6, label=v)
        plt.legend(markerscale=3, frameon=False)
    else:
        plt.scatter(emb2[:, 0], emb2[:, 1], s=4, alpha=0.6)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("Residual-space manifold (UMAP)")
    savefig(out_png)

def plot_leadtime_heatmap(
    df: pd.DataFrame,
    out_png: str | Path,
    cfg: DeepenConfig,
    value_col: str | None = None,
):
    # heatmap of density over (TTE, score or deviation magnitude)
    d = df.dropna(subset=[cfg.tte_col]).copy()

    # For lead-time structure, focus on non-controls by default if available
    if cfg.status_col in d.columns:
        d = d[d[cfg.status_col].astype(str).isin(["Prodromal", "Diagnosed"])]

    t = d[cfg.tte_col].values.astype(float)

    if value_col is None:
        value_col = cfg.score_col
    d = d.dropna(subset=[value_col])
    v = d[value_col].values.astype(float)

    # Restrict TTE window
    m = (t >= cfg.leadtime_min) & (t <= cfg.leadtime_max)
    t = t[m]
    v = v[m]

    # 2D histogram
    xbins = np.linspace(cfg.leadtime_min, cfg.leadtime_max, cfg.leadtime_bins + 1)
    ybins = np.linspace(np.percentile(v, 1), np.percentile(v, 99), 60)
    H, xe, ye = np.histogram2d(t, v, bins=[xbins, ybins])

    # Smooth + log for readability when counts are sparse
    try:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=1.0)
    except Exception:
        pass

    plt.figure(figsize=(7.0, 4.8))
    plt.imshow(
        np.log1p(H).T,
        origin="lower",
        aspect="auto",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
    )
    plt.colorbar(label="log(1 + count)")
    plt.xlabel("Years to event (negative = before diagnosis)")
    plt.ylabel(value_col)
    plt.title("Lead-time density heatmap")
    savefig(out_png)

def find_mislabeled_cohorts(
    df: pd.DataFrame,
    cfg: DeepenConfig,
) -> pd.DataFrame:
    """Define high-score but label-negative cohort.
    Assumes labels.csv includes a binary 'Label' column (ever diagnosed) and Status.
    """
    out = df.copy()
    if "Label" not in out.columns:
        raise ValueError("labels.csv must include a 'Label' column for mislabeled analysis")

    thr = out[cfg.score_col].quantile(cfg.high_score_quantile)
    out["high_score"] = out[cfg.score_col] >= thr
    out["mislabeled_fp"] = out["high_score"] & (out["Label"] == 0)
    return out

def write_deepening_outputs(
    df: pd.DataFrame,
    outdir: str | Path,
    cfg: DeepenConfig,
    notes: dict | None = None,
):
    outdir = Path(outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)

    df.to_csv(outdir / "tables" / "subject_table.csv", index=False)

    summary = {
        "n": int(len(df)),
        "n_cases": int((df.get("Label", pd.Series([0]*len(df))) == 1).sum()) if "Label" in df.columns else None,
    }
    if notes:
        summary.update(notes)

    with open(outdir / "deepening_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
