# src/genetics/ldsc.py
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import yaml
import pandas as pd

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/genetics.yaml")
    ap.add_argument("--sumstats", nargs="+", required=True, help="List of plink2 glm outputs to munge and run rg on")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    out_root = Path(cfg["out_root"]) / "ldsc"
    out_root.mkdir(parents=True, exist_ok=True)

    munge = cfg["munge_sumstats_py"]
    ldsc = cfg["ldsc_py"]
    ld_scores = cfg["ld_scores"]
    w_ld_chr = cfg["w_ld_chr"]
    hm3 = cfg["hm3_snplist"]

    munged = []
    for ss in args.sumstats:
        ss = Path(ss)
        out_prefix = out_root / ss.stem
        cmd = [
            "python", munge,
            "--sumstats", str(ss),
            "--out", str(out_prefix),
            "--merge-alleles", str(hm3),
        ]
        run(cmd)
        munged.append(str(out_prefix) + ".sumstats.gz")

    # run pairwise rg across all provided sumstats
    rg_out = out_root / "rg"
    cmd = [
        "python", ldsc,
        "--rg", ",".join(munged),
        "--ref-ld-chr", str(ld_scores),
        "--w-ld-chr", str(w_ld_chr),
        "--out", str(rg_out),
    ]
    run(cmd)

    print("[OK] LDSC rg ->", rg_out)

if __name__ == "__main__":
    main()
