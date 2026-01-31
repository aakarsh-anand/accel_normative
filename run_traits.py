# src/accel/run_traits.py
from __future__ import annotations

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


def run(cmd: List[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd)


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def collect_trait_summary(trait_dir: Path, trait_name: str) -> Dict[str, Any]:
    """
    Collect:
      - axis CV metrics: outputs/axis/pd_axis_cv_metrics.csv
      - Cox per-SD: survival/cox_<score>_perSD.csv
      - Cox tertiles: survival/cox_<score>_tertiles.csv
      - Case/control counts from axis CV metrics
    """
    out: Dict[str, Any] = {"trait": trait_name, "trait_dir": str(trait_dir)}

    axis_metrics = read_csv_if_exists(trait_dir / "axis" / "pd_axis_cv_metrics.csv")
    if axis_metrics is not None and len(axis_metrics):
        row = axis_metrics.iloc[0].to_dict()
        for k, v in row.items():
            out[f"axis_{k}"] = v

    # Cox per-SD: we want HR, CI, p for the score term
    cox_per = read_csv_if_exists(trait_dir / "survival" / "cox_pd_axis_proj_perSD.csv")
    if cox_per is not None and len(cox_per):
        # lifelines columns usually include: exp(coef), exp(coef) lower 95%, exp(coef) upper 95%, p
        # term column: "pd_axis_proj (per +1 SD)"
        # robustly pick first row
        r = cox_per.iloc[0].to_dict()
        for k in ["term", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]:
            if k in r:
                out[f"cox_perSD_{k}"] = r[k]

    # Cox tertiles: Mid vs Low, High vs Low
    cox_tert = read_csv_if_exists(trait_dir / "survival" / "cox_pd_axis_proj_tertiles.csv")
    if cox_tert is not None and len(cox_tert):
        for _, r in cox_tert.iterrows():
            term = str(r.get("term", "")).replace(" ", "_")
            for k in ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]:
                if k in r:
                    out[f"cox_tertiles_{term}_{k}"] = r[k]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits_yaml", required=True)
    ap.add_argument("--resid_key", required=True)
    ap.add_argument("--skip_plots", action="store_true")
    ap.add_argument("--skip_survival", action="store_true")
    args = ap.parse_args()

    with open(args.traits_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    g = cfg["global"]
    traits: Dict[str, Any] = cfg["traits"]

    pull_date = g["pull_date"]
    icd_csv = g["icd_csv"]
    resid_dir = g["resid_dir"]
    outroot = Path(g.get("outroot", "outputs/traits"))
    outroot.mkdir(parents=True, exist_ok=True)

    id_col_icd = g.get("id_col_icd", "participant.eid")
    accel_start_col = g.get("accel_start_col", "participant.p90003")
    score_col = g.get("score_col", "pd_axis_proj")
    km_ymin = float(g.get("km_ymin", 0.985))

    all_summaries: List[Dict[str, Any]] = []

    for trait_name, tcfg in traits.items():
        print("\n" + "=" * 80)
        print(f"TRAIT: {trait_name}")
        print("=" * 80)

        event_cols = tcfg.get("event_date_cols", [])
        if not event_cols:
            print(f"[SKIP] {trait_name}: event_date_cols empty.")
            continue

        trait_dir = outroot / trait_name
        labels_csv = trait_dir / "labels.csv"
        axis_dir = trait_dir / "axis"
        figs_dir = trait_dir / "figs"
        surv_dir = trait_dir / "survival"

        axis_dir.mkdir(parents=True, exist_ok=True)
        figs_dir.mkdir(parents=True, exist_ok=True)
        surv_dir.mkdir(parents=True, exist_ok=True)

        # 1) make labels
        run([
            sys.executable, "-m", "src.disease.make_labels",
            "--icd_csv", icd_csv,
            "--out_csv", str(labels_csv),
            "--id_col_icd", id_col_icd,
            "--accel_start_col", accel_start_col,
            "--pd_cols", ",".join(event_cols),
        ])

        # 2) fit axis
        run([
            sys.executable, "-m", "src.disease.axis",
            "--resid_dir", resid_dir,
            "--pd_labels_csv", str(labels_csv),
            "--outdir", str(axis_dir),
            "--resid_key", args.resid_key,
        ])

        # 3) plots
        if not args.skip_plots:
            run([
                sys.executable, "-m", "src.disease.plots",
                "--resid_dir", resid_dir,
                "--pd_axis_scores_csv", str(axis_dir / "pd_axis_scores.csv"),
                "--outdir", str(figs_dir),
            ])

        # 4) survival
        if not args.skip_survival:
            run([
                sys.executable, "-m", "src.disease.survival",
                "--labels_csv", str(labels_csv),
                "--score_csv", str(axis_dir / "pd_axis_scores.csv"),
                "--score_col", score_col,
                "--pull_date", pull_date,
                "--outdir", str(surv_dir),
                "--ymin", str(km_ymin),
            ])

        # Collect summary for this trait
        summ = collect_trait_summary(trait_dir, trait_name)
        all_summaries.append(summ)

        # Persist per-trait summary json for debugging
        with open(trait_dir / "summary.json", "w") as f:
            json.dump(summ, f, indent=2)

        print(f"[OK] Finished {trait_name} -> {trait_dir}")

    # Write combined summary table
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = outroot / "trait_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 80)
        print("[OK] Wrote combined summary:", summary_path)
        print(summary_df.head().to_string(index=False))
    else:
        print("[WARN] No traits completed; no summary written.")


if __name__ == "__main__":
    main()
