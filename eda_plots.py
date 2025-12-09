# eda_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def eda_overview(df):
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nNull Values:\n", df.isnull().sum())
    print("\nData Types:\n", df.dtypes)


# 1. Histogram plots
def plot_histograms(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['Transaction_Amount'], bins=50, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.histplot(df['Account_Balance'], bins=50, kde=True)
    plt.title("Account Balance Distribution")
    plt.show()


# 2. Time-series analysis
def plot_time_series(df):
    df_ts = df.copy()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'], errors='coerce')
    df_ts['Month'] = df_ts['Date'].dt.to_period('M')

    monthly = df_ts.groupby('Month').size()

    plt.figure(figsize=(10,4))
    monthly.plot(kind='line', marker='o')
    plt.title("Transactions per Month")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.show()

    daily_fraud = df_ts.groupby(df_ts['Date'].dt.date)['Fraud_Label'].sum()

    plt.figure(figsize=(12,4))
    daily_fraud.plot()
    plt.title("Daily Fraud Trend")
    plt.xticks(rotation=45)
    plt.ylabel("Fraud Count")
    plt.show()


# 3. Correlation Heatmaps
def plot_correlation(df):
    drop_columns = [
        'Transaction_ID','User_ID','Transaction_Amount_clean','Account_Balance_clean',
        'Transaction_Amount_scaled','Transaction_Type_scaled','Account_Balance_scaled',
        'Device_Type_scaled','Location_scaled','Merchant_Category_scaled',
        'Previous_Fraudulent_Activity_scaled','Daily_Transaction_Count_scaled',
        'Card_Type_scaled','Card_Age_scaled','Fraud_Label_scaled',
        'Transaction_Amount_clean_scaled','Account_Balance_clean_scaled',
        'Month','Weekday'
    ]

    drop_cols_present = [col for col in drop_columns if col in df.columns]
    df2 = df.drop(columns=drop_cols_present, errors='ignore')

    numeric_df = df2.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(14,10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(14,10))
    sns.heatmap(corr, cmap="coolwarm", annot=True, mask=mask, fmt=".2f")
    plt.title("Triangular Correlation Heatmap")
    plt.show()


def plot_fraud_distribution(df):
    plt.figure(figsize=(6,5))
    sns.countplot(x=df["Fraud_Label"])
    plt.title("Fraud vs Non-Fraud Count")
    plt.show()


def plot_amount_vs_fraud(df):
    plt.figure(figsize=(7,5))
    sns.boxplot(x="Fraud_Label", y="Transaction_Amount", data=df)
    plt.title("Amount by Fraud Status")
    plt.show()


def plot_fraud_by_device(df):
    fraud_by_device = df.groupby("Device_Type")["Fraud_Label"].mean()

    plt.figure(figsize=(8,5))
    sns.barplot(x=fraud_by_device.index, y=fraud_by_device.values)
    plt.title("Fraud Rate by Device Type")
    plt.show()


def plot_top_fraud_locations(df):
    top_fraud = df[df["Fraud_Label"] == 1].groupby("Location")["Fraud_Label"].count().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    sns.barplot(x=top_fraud.values, y=top_fraud.index)
    plt.title("Top 10 Fraud Locations")
    plt.show()


def plot_daily_volume(df):
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date'])

    daily = df2.groupby(df2['Date'].dt.date).size()

    plt.figure(figsize=(12,5))
    plt.plot(daily)
    plt.title("Daily Transaction Volume")
    plt.show()
