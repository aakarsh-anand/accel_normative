#!/usr/bin/env bash
set -euo pipefail

export ICD_CSV=/home/aakarsh/aakarsh/icd_all.csv
export RESID_DIR=outputs/residuals_v0

TRAITS_YAML="${1:-configs/traits.yaml}"

ICD_CSV="${ICD_CSV:?Set ICD_CSV env var}"
RESID_DIR="${RESID_DIR:?Set RESID_DIR env var (e.g. outputs/residuals_v0)}"

# Your wrappers (must exist)
MAKE_LABELS_SH="${MAKE_LABELS_SH:-scripts/make_pd_labels.sh}"
FIT_AXIS_SH="${FIT_AXIS_SH:-scripts/pd_axis.sh}"
PLOTS_SH="${PLOTS_SH:-scripts/pd_plots.sh}"
SURV_SH="${SURV_SH:-scripts/pd_survival.sh}"

OUTROOT="${OUTROOT:-outputs/traits}"

python - <<'PY'
import os, sys, yaml, subprocess

traits_yaml = sys.argv[1]
with open(traits_yaml, "r") as f:
    cfg = yaml.safe_load(f)

pull_date = cfg["global"]["pull_date"]
id_col_icd = cfg["global"]["id_col_icd"]
accel_start_col = cfg["global"]["accel_start_col"]
traits = cfg["traits"]

ICD_CSV = os.environ["ICD_CSV"]
RESID_DIR = os.environ["RESID_DIR"]
OUTROOT = os.environ.get("OUTROOT", "outputs/traits")

MAKE_LABELS_SH = os.environ.get("MAKE_LABELS_SH", "scripts/make_pd_labels.sh")
FIT_AXIS_SH = os.environ.get("FIT_AXIS_SH", "scripts/pd_axis.sh")
PLOTS_SH = os.environ.get("PLOTS_SH", "scripts/pd_plots.sh")
SURV_SH = os.environ.get("SURV_SH", "scripts/pd_survival.sh")

def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd)

for name, tcfg in traits.items():
    outdir = os.path.join(OUTROOT, name)
    os.makedirs(outdir, exist_ok=True)

    labels_csv = os.path.join(outdir, "labels.csv")
    axis_out = os.path.join(outdir, "axis")
    figs_out = os.path.join(outdir, "figs")
    surv_out = os.path.join(outdir, "survival")

    os.makedirs(axis_out, exist_ok=True)
    os.makedirs(figs_out, exist_ok=True)
    os.makedirs(surv_out, exist_ok=True)

    # event date columns for this trait
    cols = tcfg.get("pd_cols", [])
    if not cols:
        print(f"[SKIP] {name}: no pd_cols provided")
        continue
    pd_cols_str = ",".join(cols)

    # 1) make labels
    run([MAKE_LABELS_SH,
         "--icd_csv", ICD_CSV,
         "--out_csv", labels_csv,
         "--id_col_icd", id_col_icd,
         "--accel_start_col", accel_start_col,
         "--pd_cols", pd_cols_str])

    # 2) fit axis
    run([FIT_AXIS_SH,
         "--resid_dir", RESID_DIR,
         "--pd_labels_csv", labels_csv,
         "--outdir", axis_out])

    # 3) plots
    run([PLOTS_SH,
         "--resid_dir", RESID_DIR,
         "--pd_axis_scores_csv", os.path.join(axis_out, "pd_axis_scores.csv"),
         "--outdir", figs_out])

    # 4) survival
    run([SURV_SH,
         "--labels_csv", labels_csv,
         "--score_csv", os.path.join(axis_out, "pd_axis_scores.csv"),
         "--score_col", "pd_axis_proj",
         "--pull_date", pull_date,
         "--outdir", surv_out,
         "--ymin", "0.985"])

    print(f"\n[OK] Completed trait: {name} -> {outdir}")

PY
