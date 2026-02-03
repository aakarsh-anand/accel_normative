#!/usr/bin/env python3
"""
Extract wear dates from accelerometer CSV files to create a season covariate.

Usage:
    python extract_wear_dates.py --data_dir /path/to/csvs --output wear_dates.csv
"""

import os
import argparse
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def extract_date_from_header(csv_path: str) -> dict:
    """
    Extract participant ID and wear start date from CSV filename and header.
    
    Returns:
        dict with 'Participant ID', 'wear_start_date', 'month', 'month_sin', 'month_cos'
    """
    # Extract ID from filename: {ID}_90004_0_0.csv
    filename = Path(csv_path).name
    match = re.match(r'(\d+)_90004_0_0\.csv', filename)
    if not match:
        return None
    
    participant_id = match.group(1)
    
    # Read first column name to extract date
    # Format: "acceleration (mg) - 2014-04-10 10:00:00 - 2014-04-17 09:59:55 - ..."
    try:
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
        
        # Extract the start date (first date in the column name)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', header)
        if not date_match:
            return None
        
        date_str = date_match.group(1)
        wear_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Extract month and create cyclic features
        month = wear_date.month
        month_sin = round(float(np.sin(2 * np.pi * month / 12)), 6)
        month_cos = round(float(np.cos(2 * np.pi * month / 12)), 6)
        
        return {
            'Participant ID': participant_id,
            'wear_start_date': wear_date.strftime('%Y-%m-%d'),
            'wear_month': month,
            'wear_month_sin': month_sin,
            'wear_month_cos': month_cos,
        }
    
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract wear dates from accelerometer CSVs")
    parser.add_argument("--data_dir", required=True, help="Directory containing {ID}_90004_0_0.csv files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--pattern", default="*_90004_0_0.csv", help="Filename pattern to match")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Find all matching CSV files
    csv_files = list(data_dir.glob(args.pattern))
    print(f"Found {len(csv_files)} CSV files matching pattern '{args.pattern}'")
    
    if len(csv_files) == 0:
        raise ValueError(f"No files found in {data_dir} matching {args.pattern}")
    
    # Extract dates from all files
    results = []
    for csv_path in tqdm(csv_files, desc="Extracting dates"):
        result = extract_date_from_header(str(csv_path))
        if result:
            results.append(result)
    
    print(f"Successfully extracted dates from {len(results)} files")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values('Participant ID').reset_index(drop=True)
    
    df.to_csv(args.output, index=False)
    print(f"\n[OK] Saved to: {args.output}")
    print(f"\nPreview:")
    print(df.head(10))
    print(f"\nMonth distribution:")
    print(df['wear_month'].value_counts().sort_index())


if __name__ == "__main__":
    main()