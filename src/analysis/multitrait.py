# src/analysis/multitrait.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .viz import savefig

def load_axis_unit(trait_axis_dir: str | Path) -> np.ndarray | None:
    p = Path(trait_axis_dir) / "pd_axis_unit.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    # allow generic name too
    p2 = Path(trait_axis_dir) / "axis_unit.npy"
    if p2.exists():
        return np.load(p2).astype(np.float32)
    return None

def trait_embedding_from_axes(traits: list[dict]) -> tuple[np.ndarray, list[str]]:
    X = []
    names = []
    for t in traits:
        u = load_axis_unit(t["axis_dir"])
        if u is None:
            continue
        X.append(u.reshape(1, -1))
        names.append(t["name"])
    if not X:
        raise ValueError("No axis_unit vectors found for any trait.")
    X = np.vstack(X)
    # Normalize for cosine comparisons
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, names

def trait_embedding_from_centroids(
    resid: np.ndarray,
    resid_ids: pd.Series,
    traits: list[dict],
    id_col: str = "Participant ID",
) -> tuple[np.ndarray, list[str]]:
    X = []
    names = []
    id_to_idx = {str(i): k for k, i in enumerate(resid_ids.astype(str).values)}
    for t in traits:
        lab = pd.read_csv(t["labels_csv"])
        if "Label" not in lab.columns:
            continue
        # align indices
        idx = lab[id_col].astype(str).map(id_to_idx).dropna().astype(int).values
        y = lab.set_index(id_col).loc[lab[id_col].astype(str).iloc[:len(lab)], "Label"].values
        # safer alignment:
        lab2 = lab.copy()
        lab2["idx"] = lab2[id_col].astype(str).map(id_to_idx)
        lab2 = lab2.dropna(subset=["idx"])
        idx = lab2["idx"].astype(int).values
        y = lab2["Label"].astype(int).values

        if (y == 1).sum() < 20 or (y == 0).sum() < 200:
            continue
        mu1 = resid[idx[y == 1]].mean(axis=0)
        mu0 = resid[idx[y == 0]].mean(axis=0)
        X.append((mu1 - mu0).reshape(1, -1))
        names.append(t["name"])
    if not X:
        raise ValueError("No centroid embeddings could be computed for any trait.")
    return np.vstack(X), names

def plot_trait_umap(X: np.ndarray, names: list[str], out_png: str | Path):
    import umap
    emb2 = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=0, metric="cosine").fit_transform(X)

    plt.figure(figsize=(7.2, 5.6))
    plt.scatter(emb2[:, 0], emb2[:, 1], s=25, alpha=0.8)
    for i, name in enumerate(names):
        plt.text(emb2[i, 0], emb2[i, 1], name, fontsize=8)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("Trait atlas (UMAP on trait embeddings)")
    savefig(out_png)

def cluster_traits(X: np.ndarray) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    # With small numbers of traits, fixed small k is more stable than a threshold.
    k = min(4, max(2, X.shape[0] // 4))
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
