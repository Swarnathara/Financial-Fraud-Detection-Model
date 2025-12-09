# data_loader.py
import pandas as pd
import numpy as np

DEFAULT_PATH = r"D:\amdox\Project_2 Data analytics\project_2\data\preprocessed_synthetic_fraud_data.csv"

def load_data(path: str = DEFAULT_PATH):
    """Load data for both EDA and ML."""
    df = pd.read_csv(path)
    print("[INFO] Data Loaded:", df.shape)
    return df


def get_ml_dataset(df):
    """
    Prepare dataset for ML prediction or training.

    For training:
        - Fraud_Label must be present.

    For streaming prediction:
        - Fraud_Label may be missing.
    """

    required_cols = [
        "Transaction_Amount",
        "Transaction_Type",
        "Account_Balance",
        "Device_Type",
        "Location",
        "Merchant_Category",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Card_Type",
        "Card_Age"
    ]

    # Make sure all ML feature columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ML columns: {missing}")

    # If Fraud_Label exists → keep it, else create dummy
    if "Fraud_Label" in df.columns:
        df_ml = df[required_cols + ["Fraud_Label"]].copy()
    else:
        df_ml = df[required_cols].copy()

    return df_ml


if __name__ == "__main__":
    df = load_data()
    print("\nEDA DATAFRAME (all columns):")
    print(df.head())

    ml_df = get_ml_dataset(df)
    print("\nML DATAFRAME (ML-ready):")
    print(ml_df.head())
