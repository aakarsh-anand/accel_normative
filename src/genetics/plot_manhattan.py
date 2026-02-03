# src/genetics/plot_manhattan.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.viz import savefig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumstats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p_col", default="P")
    ap.add_argument("--chr_col", default="CHR")
    ap.add_argument("--bp_col", default="BP")
    args = ap.parse_args()

    df = pd.read_csv(args.sumstats, sep=None, engine="python")
    for c in [args.p_col, args.chr_col, args.bp_col]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in columns: {list(df.columns)}")

    df = df.dropna(subset=[args.p_col, args.chr_col, args.bp_col]).copy()
    df[args.chr_col] = df[args.chr_col].astype(int)
    df[args.bp_col] = df[args.bp_col].astype(int)
    df["mlogp"] = -np.log10(df[args.p_col].astype(float).clip(1e-300, 1.0))

    df = df.sort_values([args.chr_col, args.bp_col])

    # Build cumulative positions
    chr_sizes = df.groupby(args.chr_col)[args.bp_col].max()
    offsets = chr_sizes.cumsum().shift(1).fillna(0).to_dict()
    df["pos"] = df.apply(lambda r: r[args.bp_col] + offsets[r[args.chr_col]], axis=1)

    plt.figure(figsize=(12, 4.2))
    plt.scatter(df["pos"], df["mlogp"], s=2, alpha=0.8)
    plt.axhline(-np.log10(5e-8), linestyle="--")
    plt.ylabel("-log10(P)")
    plt.xlabel("Genomic position")
    plt.title("Manhattan plot")
    savefig(args.out)

if __name__ == "__main__":
    main()
