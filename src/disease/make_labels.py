# src/accel/make_labels.py
from __future__ import annotations

import os
import argparse
import pandas as pd

from .labels import PDLabelConfig, compute_pd_tte_and_status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icd_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_col_icd", default="participant.eid")
    ap.add_argument("--accel_start_col", default="participant.p90003")
    ap.add_argument("--pd_cols", required=True,
                    help="Comma-separated ICD date columns that indicate PD diagnosis (datetimes)")
    args = ap.parse_args()

    icd = pd.read_csv(args.icd_csv)
    pd_cols = [c.strip() for c in args.pd_cols.split(",") if c.strip()]

    cfg = PDLabelConfig(
        id_col_icd=args.id_col_icd,
        accel_start_col=args.accel_start_col,
        pd_date_cols=pd_cols
    )
    out = compute_pd_tte_and_status(icd, cfg)
    out.to_csv(args.out_csv, index=False)
    print("[OK] wrote", args.out_csv)
    print(out["Status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
