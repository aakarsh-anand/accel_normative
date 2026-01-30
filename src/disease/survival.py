# src/accel/survival.py
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts


def ensure_dir(d): os.makedirs(d, exist_ok=True)


def make_survival_table(
    labels: pd.DataFrame,
    pull_date: str,
    score_df: pd.DataFrame,
    score_col: str,
) -> pd.DataFrame:
    """
    labels must have:
      Participant ID, accel_start_date, pd_date
    score_df must have:
      Participant ID, score_col
    pull_date: YYYY-MM-DD (censor at pull date if no event)
    """
    df = labels.copy()
    df["Participant ID"] = df["Participant ID"].astype(str)
    df["accel_start_date"] = pd.to_datetime(df["accel_start_date"], errors="coerce")
    df["pd_date"] = pd.to_datetime(df["pd_date"], errors="coerce")
    censor_date = pd.to_datetime(pull_date)

    df["event"] = df["pd_date"].notna().astype(int)

    end_date = df["pd_date"].fillna(censor_date)
    duration_years = (end_date - df["accel_start_date"]).dt.total_seconds() / (365.25 * 24 * 3600)
    df["duration_years"] = pd.to_numeric(duration_years, errors="coerce")

    s = score_df[["Participant ID", score_col]].copy()
    s["Participant ID"] = s["Participant ID"].astype(str)
    df = df.merge(s, on="Participant ID", how="left")

    df = df.dropna(subset=["duration_years", score_col])
    df = df[df["duration_years"] >= 0]
    return df


def tertile_groups(x: pd.Series) -> pd.Series:
    q1, q2 = x.quantile([1/3, 2/3]).values
    def grp(v):
        if v <= q1: return "Low"
        if v <= q2: return "Mid"
        return "High"
    return x.apply(grp)


def km_plot_with_risktable(df: pd.DataFrame, score_col: str, outpath: str, ymin: float = 0.985):
    df = df.copy()
    df["group"] = tertile_groups(df[score_col])

    # Taller figure so risk table fits cleanly
    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    kmfs = {}
    for g in ["Low", "Mid", "High"]:
        sub = df[df["group"] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["duration_years"], event_observed=sub["event"], label=g)
        kmf.plot_survival_function(ax=ax, ci_show=True)
        kmfs[g] = kmf

    ax.set_title(f"Kaplan–Meier: incident outcome by tertiles of {score_col}")
    ax.set_xlabel("Years since accelerometer wear")
    ax.set_ylabel("Survival probability (not diagnosed)")
    ax.set_ylim(ymin, 1.0005)

    # Put legend in a clean spot
    ax.legend(frameon=True, loc="lower left")

    # IMPORTANT: Show only the rows we want (avoid huge table)
    # Options: ["At risk"] or ["At risk", "Events"]
    add_at_risk_counts(
        kmfs["Low"], kmfs["Mid"], kmfs["High"],
        ax=ax,
        rows_to_show=["At risk"],   # <-- make compact
        fontsize=10
    )

    # Add space at bottom for the table
    fig.subplots_adjust(bottom=0.25)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def cox_continuous(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Cox on z-scored score. Reports HR per +1 SD.
    """
    d = df.copy()
    x = d[score_col].astype(float)
    d[f"{score_col}_z"] = (x - x.mean()) / (x.std(ddof=0) + 1e-12)

    cph = CoxPHFitter()
    cph.fit(
        d[[f"{score_col}_z", "duration_years", "event"]],
        duration_col="duration_years",
        event_col="event",
    )

    summ = cph.summary.reset_index()

    # lifelines may name this column 'covariate' or 'index'
    if "covariate" in summ.columns:
        summ = summ.rename(columns={"covariate": "term"})
    elif "index" in summ.columns:
        summ = summ.rename(columns={"index": "term"})
    elif "term" not in summ.columns:
        # fallback: first column after reset_index is usually the term
        summ = summ.rename(columns={summ.columns[0]: "term"})

    summ.loc[summ["term"] == f"{score_col}_z", "term"] = f"{score_col} (per +1 SD)"
    return summ

def cox_tertiles(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Cox with tertile groups (Low as reference) -> HR Mid vs Low, High vs Low.
    """
    d = df.copy()
    d["group"] = tertile_groups(d[score_col])
    # one-hot encode with Low as reference
    d["group_Mid"] = (d["group"] == "Mid").astype(int)
    d["group_High"] = (d["group"] == "High").astype(int)

    cph = CoxPHFitter()
    cph.fit(d[["group_Mid", "group_High", "duration_years", "event"]],
            duration_col="duration_years", event_col="event")
    summ = cph.summary.reset_index()
    
    # lifelines may name this column 'covariate' or 'index'
    if "covariate" in summ.columns:
        summ = summ.rename(columns={"covariate": "term"})
    elif "index" in summ.columns:
        summ = summ.rename(columns={"index": "term"})
    elif "term" not in summ.columns:
        # fallback: first column after reset_index is usually the term
        summ = summ.rename(columns={summ.columns[0]: "term"})

    summ["term"] = summ["term"].replace({"group_Mid": "Mid vs Low", "group_High": "High vs Low"})
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True,
                    help="CSV with Participant ID, accel_start_date, pd_date")
    ap.add_argument("--score_csv", required=True,
                    help="CSV containing Participant ID and score column")
    ap.add_argument("--score_col", default="pd_axis_proj")
    ap.add_argument("--pull_date", required=True, help="YYYY-MM-DD label pull date for censoring")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ymin", type=float, default=0.985, help="KM y-axis lower bound (rare events)")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    labels = pd.read_csv(args.labels_csv)
    score_df = pd.read_csv(args.score_csv)

    df = make_survival_table(labels, args.pull_date, score_df, args.score_col)
    df.to_csv(os.path.join(args.outdir, "survival_table.csv"), index=False)

    # KM + risk table
    km_plot_with_risktable(df, args.score_col, os.path.join(args.outdir, f"km_{args.score_col}.png"), ymin=args.ymin)

    # Cox: continuous z-scored
    cont = cox_continuous(df, args.score_col)
    cont.to_csv(os.path.join(args.outdir, f"cox_{args.score_col}_perSD.csv"), index=False)

    # Cox: tertiles
    tert = cox_tertiles(df, args.score_col)
    tert.to_csv(os.path.join(args.outdir, f"cox_{args.score_col}_tertiles.csv"), index=False)

    print("[OK] wrote survival outputs to", args.outdir)
    print("\nCox per-SD:\n", cont.to_string(index=False))
    print("\nCox tertiles:\n", tert.to_string(index=False))


if __name__ == "__main__":
    main()
