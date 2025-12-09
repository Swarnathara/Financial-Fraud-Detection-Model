import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Path to your main dataset
DATASET_PATH = r"D:\amdox\Project_2 Data analytics\project_2\data\preprocessed_synthetic_fraud_data.csv"

# Folder where streaming engine is watching
STREAM_INPUT = r"D:\amdox\Project_2 Data analytics\project_2\fraud_streaming\stream_input"

# Number of CSV files to generate
NUM_FILES = 10

# Min/max rows per generated file
ROWS_MIN = 1
ROWS_MAX = 5

def generate_stream_files():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print("[INFO] Dataset loaded:", df.shape)

    os.makedirs(STREAM_INPUT, exist_ok=True)

    for i in range(1, NUM_FILES + 1):
        rows_to_pick = np.random.randint(ROWS_MIN, ROWS_MAX + 1)
        sample_df = df.sample(rows_to_pick)

        file_name = f"auto_{i}.csv"
        file_path = os.path.join(STREAM_INPUT, file_name)

        sample_df.to_csv(file_path, index=False)
        
        print(f"[GENERATED] {file_name} with {rows_to_pick} rows")

        time.sleep(1)   # small delay between file creation

    print("\n[INFO] Completed generating files.")
    print("[INFO] You can now run your stream engine and Streamlit dashboard.\n")


if __name__ == "__main__":
    generate_stream_files()
