# auto_generate_stream.py
import pandas as pd
import os
import time
from datetime import datetime

# ---------------- CONFIG ----------------
INPUT_CSV = "preprocessed_synthetic_fraud_data.csv"  # Your dataset
OUTPUT_DIR = "fraud_streaming/stream_input"          # Where new CSVs will go
ROWS_PER_FILE = 5                                    # Number of rows per CSV
FREQUENCY_SEC = 5                                    # Interval between CSVs
TOTAL_FILES = 5                                      # Total number of CSVs to generate
# ----------------------------------------

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the full dataset once
df = pd.read_csv(INPUT_CSV)

print(f"[INFO] Loaded dataset with {len(df)} rows.")
print(f"[INFO] Auto-generating {TOTAL_FILES} CSVs every {FREQUENCY_SEC} seconds...")

for i in range(TOTAL_FILES):
    # Pick random rows
    sample_df = df.sample(n=ROWS_PER_FILE, replace=True)

    # Generate timestamped file name
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"test_{timestamp}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Save CSV
    sample_df.to_csv(filepath, index=False)
    print(f"[INFO] Generated {filename} with {ROWS_PER_FILE} rows ({i+1}/{TOTAL_FILES}).")

    # Wait before next file, except after the last one
    if i < TOTAL_FILES - 1:
        time.sleep(FREQUENCY_SEC)

print("\n[INFO] Auto-generation completed. Total files generated:", TOTAL_FILES)
