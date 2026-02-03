# src/genetics/gwas_plink2.py
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import yaml

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/genetics.yaml")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--kind", choices=["binary", "continuous", "control_continuous"], required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    out_root = Path(cfg["out_root"])
    pheno = out_root / "phenotypes" / args.trait / f"{args.kind}.tsv"
    outdir = out_root / "gwas" / args.trait / args.kind
    outdir.mkdir(parents=True, exist_ok=True)

    plink2 = cfg["plink2"]
    geno = cfg["genotype_prefix"]
    covar = cfg["covar_csv"]
    covar_cols = cfg["covar_cols"]

    out_prefix = outdir / "plink2"

    cmd = [
        plink2,
        "--pfile", geno,
        "--pheno", str(pheno),
        "--pheno-name", "PHENO",
        "--covar", str(covar),
        "--covar-name", ",".join(covar_cols),
        "--threads", str(args.threads),
        "--out", str(out_prefix),
    ]

    # Model choice
    if args.kind == "binary":
        cmd += ["--glm", "firth-fallback"]
    else:
        cmd += ["--glm"]

    run(cmd)
    print("[OK] GWAS:", args.trait, args.kind, "->", outdir)

if __name__ == "__main__":
    main()
