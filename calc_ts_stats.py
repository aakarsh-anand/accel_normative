import os
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_sample_from_file(file_path, sample_size=5000):
    """Reads a file and returns a random subsample of the first column."""
    try:
        # Use low_memory and only read the first column for speed
        df = pd.read_csv(file_path, usecols=[0], nrows=None, engine='c')
        data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
        
        if len(data) > sample_size:
            return np.random.choice(data, size=sample_size, replace=False)
        return data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.array([])

def calculate_global_stats(data_dir, extension="*.csv", max_workers=None):
    files = glob.glob(os.path.join(data_dir, extension))
    print(f"Found {len(files)} files. Starting subsampling...")

    all_samples = []
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Wrap with tqdm for progress tracking
        results = list(tqdm(executor.map(get_sample_from_file, files), total=len(files)))
    
    # Flatten results
    combined_data = np.concatenate([r for r in results if r.size > 0])
    
    print("Computing final statistics...")
    median = np.median(combined_data)
    q25 = np.percentile(combined_data, 25)
    q75 = np.percentile(combined_data, 75)
    iqr = q75 - q25
    
    return {
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "total_samples": len(combined_data)
    }

if __name__ == "__main__":
    # Change this to your directory path
    ACCEL_DIR = "/home/aakarsh/accelerometer/ts/"
    
    stats = calculate_global_stats(ACCEL_DIR)
    
    print("\n--- Global Statistics ---")
    print(f"Median: {stats['median']:.4f}")
    print(f"Q25:    {stats['q25']:.4f}")
    print(f"Q75:    {stats['q75']:.4f}")
    print(f"IQR:    {stats['iqr']:.4f}")
    print(f"Based on {stats['total_samples']} data points.")