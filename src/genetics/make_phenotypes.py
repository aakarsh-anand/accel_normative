# src/genetics/make_phenotypes.py
from __future__ import annotations
import argparse
from pathlib import Path
import json
import yaml
import pandas as pd

from .phenotypes import (
    load_labels, load_axis_scores, make_binary_pheno, make_continuous_pheno,
    make_control_replaced_pheno, to_plink_pheno
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/genetics.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    traits_yaml = cfg["traits_yaml"]
    traits_root = Path(cfg["traits_root"])
    resid_summary = pd.read_csv(cfg["residual_summary_csv"])
    id_col = cfg["id_col"]
    ids = resid_summary[id_col].astype(str)

    trait_cfg = yaml.safe_load(open(traits_yaml, "r"))
    trait_names = [t["name"] for t in trait_cfg.get("traits", [])]

    out_root = Path(cfg["out_root"]) / "phenotypes"
    out_root.mkdir(parents=True, exist_ok=True)

    for name in trait_names:
        lab_path = traits_root / name / "labels.csv"
        scores_path = traits_root / name / cfg["axis_scores_relpath"]

        if not lab_path.exists() or not scores_path.exists():
            continue

        labels = load_labels(lab_path, id_col=id_col)
        scores = load_axis_scores(scores_path, id_col=id_col, score_col_candidates=cfg["axis_score_col_candidates"])

        outdir = out_root / name
        outdir.mkdir(parents=True, exist_ok=True)

        bin_df = to_plink_pheno(make_binary_pheno(ids, labels, id_col), id_col)
        cont_df = to_plink_pheno(make_continuous_pheno(ids, scores, id_col), id_col)
        ctrl_cont_df = to_plink_pheno(make_control_replaced_pheno(ids, labels, scores, id_col), id_col)

        bin_df.to_csv(outdir / "binary.tsv", sep="\t", index=False)
        cont_df.to_csv(outdir / "continuous.tsv", sep="\t", index=False)
        ctrl_cont_df.to_csv(outdir / "control_continuous.tsv", sep="\t", index=False)

        manifest = {
            "trait": name,
            "labels_csv": str(lab_path),
            "axis_scores_csv": str(scores_path),
            "n_total": int(len(ids)),
            "n_cases": int(labels["Label"].sum()),
        }
        json.dump(manifest, open(outdir / "manifest.json", "w"), indent=2)

        print("[OK] phenotypes:", name)

if __name__ == "__main__":
    main()
