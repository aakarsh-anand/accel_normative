# src/genetics/plot_rg.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.viz import savefig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rg_log", required=True, help="LDSC rg output .log or .results file")
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    # LDSC writes rg results in a .log; easiest is to also locate .results if present
    p = Path(args.rg_log)
    results = p.with_suffix(".results")
    if results.exists():
        df = pd.read_csv(results, sep="\t")
    else:
        # fallback: try reading log-like table if user points directly to .results
        df = pd.read_csv(p, sep="\t")

    # Expect columns: p1, p2, rg, se, p
    need = ["p1", "p2", "rg"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing {c}. Columns: {list(df.columns)}")

    # Build matrix
    traits = sorted(set(df["p1"]).union(set(df["p2"])))
    idx = {t: i for i, t in enumerate(traits)}
    M = np.full((len(traits), len(traits)), np.nan)

    for _, r in df.iterrows():
        i, j = idx[r["p1"]], idx[r["p2"]]
        M[i, j] = r["rg"]
        M[j, i] = r["rg"]
    np.fill_diagonal(M, 1.0)

    plt.figure(figsize=(8, 7))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="rg")
    plt.xticks(range(len(traits)), traits, rotation=90, fontsize=7)
    plt.yticks(range(len(traits)), traits, fontsize=7)
    plt.title("Genetic correlation (LDSC)")
    savefig(args.out_png)

if __name__ == "__main__":
    main()
