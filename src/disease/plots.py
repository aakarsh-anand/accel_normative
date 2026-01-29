# src/accel/pd_plots.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

PALETTE = {
    "Control": "#1f77b4",
    "Prodromal": "#ff7f0e",
    "Diagnosed": "#2ca02c",
}

def ensure_dir(d): os.makedirs(d, exist_ok=True)


def make_tte_bins(tte: pd.Series, edges_years: list[float]) -> pd.Categorical:
    # edges like [0,1,2,5,10]
    edges = [-np.inf] + edges_years + [np.inf]
    labels = []
    for i in range(len(edges)-1):
        a, b = edges[i], edges[i+1]
        if a == -np.inf:
            labels.append(f"<= {b:g}")
        elif b == np.inf:
            labels.append(f"> {a:g}")
        else:
            labels.append(f"({a:g}, {b:g}]")
    return pd.cut(tte, bins=edges, labels=labels, include_lowest=True)


def boxplot_status(df, ycol, outpath, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = ["Control","Prodromal","Diagnosed"]
    sns.boxplot(
        data=df, x="Status", y=ycol, order=order,
        palette=[PALETTE[o] for o in order],
        showfliers=False, ax=ax
    )
    sns.stripplot(
        data=df.sample(min(len(df), 5000), random_state=0),
        x="Status", y=ycol, order=order,
        color="k", alpha=0.15, size=2, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def multipanel_tte(df, ycol, outpath, edges=(0.5,1,2,5,10)):
    """
    Compact grid: one panel per TTE bin (prodromals only), each panel shows Control vs that bin.
    Diagnosed shown separately in its own panel.
    """
    df = df.copy()
    df["TTE_bin"] = make_tte_bins(df["TTE_years"], list(edges))

    # bins for prodromal only
    prod = df[df["Status"] == "Prodromal"].copy()
    bins = [b for b in prod["TTE_bin"].dropna().unique().tolist()]
    bins = sorted(bins, key=lambda x: str(x))

    panels = [("Diagnosed", None)] + [(str(b), b) for b in bins]

    n = len(panels)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.0 * rows), squeeze=False)
    axes = axes.flatten()

    for i, (name, b) in enumerate(panels):
        ax = axes[i]

        if name == "Diagnosed":
            sub = df[df["Status"].isin(["Control","Diagnosed"])].copy()
            sub["Group"] = sub["Status"]
            order = ["Control","Diagnosed"]
        else:
            sub = pd.concat([
                df[df["Status"] == "Control"],
                prod[prod["TTE_bin"] == b]
            ], axis=0)
            sub = sub.copy()
            sub["Group"] = sub["Status"].astype(str) + " " + str(b)
            # keep colors: control vs prodromal
            order = ["Control","Prodromal"]
            # we'll plot Status directly for consistent palette
            sub["Status2"] = sub["Status"]
            order_status = ["Control","Prodromal"]

        if name == "Diagnosed":
            sns.boxplot(
                data=sub, x="Status", y=ycol, order=order,
                palette=[PALETTE[o] for o in order],
                showfliers=False, ax=ax
            )
            ax.set_title("Diagnosed vs Control")
        else:
            sns.boxplot(
                data=sub, x="Status2", y=ycol, order=order_status,
                palette=[PALETTE[o] for o in order_status],
                showfliers=False, ax=ax
            )
            ax.set_title(f"Prodromal {b} vs Control")

        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("PD-axis score by time-to-diagnosis bins (each panel compares to controls)", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resid_dir", required=True, help="outputs/residuals_v*/")
    ap.add_argument("--pd_axis_scores_csv", required=True, help="outputs/pd_axis_v*/pd_axis_scores.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Residual summary has resid_norm + covars; pd_axis_scores has Status/TTE + pd_axis_proj
    resid_sum = pd.read_csv(os.path.join(args.resid_dir, "residual_summary.csv"))
    pd_axis = pd.read_csv(args.pd_axis_scores_csv)

    # Merge
    df = resid_sum.merge(pd_axis[["Participant ID","Status","TTE_years","Label","pd_axis_proj","pd_axis_prob_oof"]],
                         on="Participant ID", how="left")

    df["Status"] = pd.Categorical(df["Status"], categories=["Control","Prodromal","Diagnosed"], ordered=True)

    # A) deviation magnitude
    boxplot_status(df.dropna(subset=["Status","resid_norm"]), "resid_norm",
                   os.path.join(args.outdir, "residual_norm_by_status.png"),
                   "Deviation magnitude ||r|| by PD status (age/sex/activity-adjusted)")

    # B) PD-axis projection
    boxplot_status(df.dropna(subset=["Status","pd_axis_proj"]), "pd_axis_proj",
                   os.path.join(args.outdir, "pd_axis_proj_by_status.png"),
                   "PD-axis projection (from residual embeddings) by PD status")

    # C) multipanel by TTE bins
    multipanel_tte(df.dropna(subset=["Status","pd_axis_proj"]), "pd_axis_proj",
                   os.path.join(args.outdir, "pd_axis_proj_by_tte_bins.png"),
                   edges=(0.5, 1, 2, 5, 10))


if __name__ == "__main__":
    main()
