# python_fraud_stream.py
import time
import os
import sys
import pandas as pd
import joblib
import json
import logging

# ----------------------------------------
# FIX: Add project root to import path
# ----------------------------------------
BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE)

from data_loader import get_ml_dataset

# ----------------------------------------------------
# FILE & FOLDER PATHS (CORRECTED FOR YOUR STRUCTURE)
# ----------------------------------------------------
IN_DIR = os.path.join(BASE, "fraud_streaming", "stream_input")
OUT_DIR = os.path.join(BASE, "fraud_streaming", "stream_output")
PROCESSED_DIR = os.path.join(BASE, "fraud_streaming", "processed")
ERROR_DIR = os.path.join(PROCESSED_DIR, "errors")

MODEL_PATH = os.path.join(BASE, "best_fraud_model.pkl")
SCALER_PATH = os.path.join(BASE, "fraud_scaler.pkl")
FEATURES_PATH = os.path.join(BASE, "feature_columns.json")

# ----------------------------------------------------
# Directory creation
# ----------------------------------------------------
os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

POLL_INTERVAL = 3  # seconds

# ----------------------------------------------------
# Load model + scaler + feature list
# ----------------------------------------------------
logging.info("[INFO] Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_ORDER = json.load(f)

logging.info(f"[INFO] Model & scaler loaded. Scaler expects feature names ({len(FEATURE_ORDER)}): {FEATURE_ORDER}")

seen = set()

def process_file(path):
    fname = os.path.basename(path)
    try:
        logging.info(f"[INFO] Processing file: {fname}")
        df_raw = pd.read_csv(path)

        df_ml = get_ml_dataset(df_raw)  

        X = df_ml.copy()

        if "Fraud_Label" in X.columns:
            X = X.drop(columns=["Fraud_Label"])

        missing = [c for c in FEATURE_ORDER if c not in X.columns]
        if missing:
            raise ValueError(f"Incoming file is missing features: {missing}")

        X = X[FEATURE_ORDER]
        Xs = scaler.transform(X)

        preds = model.predict(Xs)

        df_out = df_raw.copy()
        df_out["Fraud_Prediction"] = preds

        # ----------------------------------------------
        # SEND ALERT EMAIL FOR FRAUDULENT TRANSACTIONS
        # ----------------------------------------------
        from alerts import send_fraud_alert
        fraud_rows = df_out[df_out["Fraud_Prediction"] == 1]

        for _, row in fraud_rows.iterrows():
            send_fraud_alert(row)

        # ----------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------
        out_path = os.path.join(OUT_DIR, f"pred_{int(time.time())}_{fname}")
        df_out.to_csv(out_path, index=False)
        logging.info(f"[INFO] Wrote predictions to {out_path}")

        processed_name = os.path.join(PROCESSED_DIR, f"processed_{int(time.time())}_{fname}")
        os.replace(path, processed_name)
        logging.info(f"[INFO] Moved processed input to {processed_name}")

    except Exception as e:
        logging.exception(f"[ERROR] Processing failed for {fname}: {e}")
        err_name = os.path.join(ERROR_DIR, f"ERROR_{int(time.time())}_{fname}")
        try:
            os.replace(path, err_name)
            logging.info(f"[INFO] Moved problematic file to {err_name}")
        except:
            logging.warning("Could not move problematic file.")

def poll_loop():
    logging.info(f"[INFO] Starting folder watcher on: {IN_DIR}")
    while True:
        try:
            files = sorted([os.path.join(IN_DIR, f) for f in os.listdir(IN_DIR) if f.lower().endswith(".csv")])
            for p in files:
                if p not in seen:
                    seen.add(p)
                    process_file(p)
        except Exception as e:
            logging.exception("Watcher loop error: %s", e)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    poll_loop()
