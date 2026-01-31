# src/analysis/window_trajectory.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def load_window_embeddings(emb_windows_npz: str | Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(emb_windows_npz)
    # Expect keys: 'emb' and 'ids' OR similar; adjust in runner if needed.
    if "emb" in npz.files and "ids" in npz.files:
        return npz["emb"].astype(np.float32), npz["ids"].astype(str)
    raise KeyError(f"Unexpected keys in {emb_windows_npz}: {npz.files}")

def score_windows_with_model(Xw: np.ndarray, model_path: str | Path) -> np.ndarray:
    mdl = joblib.load(model_path)
    # handle possible payload formats (raw sklearn model or dict)
    if isinstance(mdl, dict) and "model" in mdl:
        mdl = mdl["model"]
    # Score as decision function if available; else predict_proba
    if hasattr(mdl, "decision_function"):
        return mdl.decision_function(Xw).astype(np.float32)
    if hasattr(mdl, "predict_proba"):
        return mdl.predict_proba(Xw)[:, 1].astype(np.float32)
    return mdl.predict(Xw).astype(np.float32)

def make_window_score_table(
    window_scores: np.ndarray,
    window_ids: np.ndarray,
    out_csv: str | Path,
    id_col: str = "Participant ID",
) -> None:
    df = pd.DataFrame({id_col: window_ids, "window_score": window_scores})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
