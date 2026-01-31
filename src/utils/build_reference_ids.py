# src/utils/build_reference_ids.py

import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--icd_csv", required=True)
    parser.add_argument("--accel_start_col", default="participant.p90003")
    parser.add_argument("--id_col", default="participant.eid")
    parser.add_argument("--buffer_years", type=float, default=5.0)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.icd_csv, parse_dates=True)

    id_col = args.id_col
    accel_col = args.accel_start_col

    accel_start = pd.to_datetime(df[accel_col])

    # Identify ICD date columns
    icd_date_cols = [
        c for c in df.columns
        if c.startswith("participant.p") and c != accel_col
    ]

    # Compute earliest ICD date per subject
    icd_dates = df[icd_date_cols].apply(pd.to_datetime, errors="coerce")
    earliest_icd = icd_dates.min(axis=1)

    buffer_delta = pd.to_timedelta(args.buffer_years * 365.25, unit="D")

    is_control = (
        earliest_icd.isna() |
        (earliest_icd > accel_start + buffer_delta)
    )

    control_ids = df.loc[is_control, id_col]

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    control_ids.to_frame('Participant ID').to_csv(out, index=False)

    print(f"Saved {len(control_ids)} reference IDs to {out}")

if __name__ == "__main__":
    main()
