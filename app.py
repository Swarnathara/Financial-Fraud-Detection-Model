# app.py
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Financial Fraud Detection System",
    layout="wide",
)

# ============================================
# PREMIUM DARK THEME CSS
# ============================================
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2933 0, #020309 45%, #000000 100%);
        color: #e5e7eb;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #060913;
        border-right: 1px solid rgba(148,163,184,0.25);
    }
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e5e7eb;
        padding-bottom: 0.5rem;
    }
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #9ca3af;
        padding-bottom: 1rem;
    }

    /* Card */
    .card {
        background: radial-gradient(circle at top left, #111827, #020617);
        border-radius: 1.25rem;
        padding: 1.5rem 1.75rem;
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 18px 45px rgba(15,23,42,0.75);
        margin-bottom: 1.5rem;
    }

    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #0f172a, #020617);
        border-radius: 1rem;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(148,163,184,0.38);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e5e7eb;
    }

    hr {
        border-color: rgba(75,85,99,0.6);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_synthetic_fraud_data.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "final_results.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
STREAM_OUT_DIR = os.path.join(BASE_DIR, "fraud_streaming", "stream_output")

# ============================================
# HELPERS
# ============================================
@st.cache_data
def load_preprocessed_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

@st.cache_data
def load_results_df():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame()

def load_stream_outputs():
    if not os.path.isdir(STREAM_OUT_DIR):
        return pd.DataFrame()

    files = [
        os.path.join(STREAM_OUT_DIR, f)
        for f in os.listdir(STREAM_OUT_DIR)
        if f.lower().endswith(".csv")
    ]
    if not files:
        return pd.DataFrame()

    dfs = []
    for fp in sorted(files):
        try:
            df = pd.read_csv(fp)
            df["__source_file"] = os.path.basename(fp)
            dfs.append(df)
        except:
            pass

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def highlight_fraud(row):
    return ["background-color: #450a0a" if row.get("Fraud_Prediction", 0) == 1 else ""] * len(row)

# ============================================
# SIDEBAR NAVIGATION (NO ICONS)
# ============================================
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

MENU_ITEMS = [
    "Overview",
    "Preprocessed Data",
    "EDA",
    "ML Model Results",
    "ML Plots",
    "Real-Time Monitoring",
]

with st.sidebar:
    st.markdown('<div class="sidebar-title"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle"><h1>MENU</h1></div>', unsafe_allow_html=True)

    for label in MENU_ITEMS:
        btn = st.button(label, use_container_width=True, key=f"nav_{label}")
        if btn:
            st.session_state["active_page"] = label
            st.rerun()

# ============================================
# PAGE FUNCTIONS
# ============================================

def page_overview():
    st.markdown("<h1>Financial Fraud Detection System</h1>", unsafe_allow_html=True)

    # Main layout: text on left, image on right
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown(
            """
            <div class="card">
               <h3>Project Overview</h3>
                <p style="text-align: justify;">
                This project implements an end-to-end <b>Real-Time Financial Fraud Detection System</b> designed to 
                identify suspicious transactions the moment they occur. The pipeline begins with extensive 
                preprocessing, which includes cleaning raw financial records, encoding categoricals, scaling numerical 
                features, and addressing class imbalance with SMOTE/SMOTE-ENN. Detailed EDA was performed to analyze 
                transaction behavior and detect hidden patterns between genuine and fraudulent records. Multiple 
                machine learning models—Logistic Regression, Random Forest, XGBoost, LightGBM—were trained and evaluated 
                using F1-score, Recall, ROC-AUC, and confusion matrices, and the best-performing model was exported 
                for deployment.
                <br><br>
                A Python-based streaming engine simulates real-world banking transactions in real time. Each 
                transaction is instantly fed to the deployed model, and predictions are written as streaming CSV logs. 
                The Streamlit dashboard monitors these files live, updating the UI every few seconds to display new 
                fraud predictions, analytics, and trends. Whenever a high-risk transaction is detected, the system 
                highlights it visually and simultaneously sends an automated email alert. This demonstrates how modern 
                financial institutions use real-time machine learning pipelines to monitor fraud continuously and 
                prevent financial loss.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        
    # Right side image
    with right:
        
        st.image(
            "1685943065092.jpg",
            use_container_width=True,
            caption="Fraud Detection — Real-Time Risk Monitoring",
        )
        st.markdown(
            """
            <div class="card">
                <h3>Team Members</h3>
                <ul>
                    <li><b>Swarnathara Ramesh</b></li>
                    <li><b>Akash Kar Choudary</b></li>
                    <li><b>Rajnandani Godage</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_preprocessed_data():
    st.markdown("<h2>Preprocessed Transaction Data</h2>", unsafe_allow_html=True)

    
    # --- TITLE FOR ENCODING CARDS ---
    # ==========================
    #      ENCODING CARDS (3 on top, 2 on bottom)
    # ==========================

    st.markdown("<h3 style='margin-top:25px;'>Categorical → Numerical Encoding</h3>", unsafe_allow_html=True)

    # ---------- FIRST ROW (3 cards) ----------
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown(
            """
            <div class="card">
                <h4>Device Type Encoding</h4>
                <ul>
                    <li>Laptop → 0</li>
                    <li>Mobile → 1</li>
                    <li>Tablet → 2</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r1c2:
        st.markdown(
            """
            <div class="card">
                <h4>Location Encoding</h4>
                <ul>
                    <li>London → 0</li>
                    <li>Mumbai → 1</li>
                    <li>New York → 2</li>
                    <li>Sydney → 3</li>
                    <li>Tokyo → 4</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r1c3:
        st.markdown(
            """
            <div class="card">
                <h4>Card Type Encoding</h4>
                <ul>
                    <li>Amex → 0</li>
                    <li>Discover → 1</li>
                    <li>MC → 2</li>
                    <li>Visa → 3</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- SECOND ROW (2 wide cards) ----------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown(
            """
            <div class="card">
                <h4>Merchant Category Encoding</h4>
                <ul>
                    <li>Clothing → 0</li>
                    <li>Electronics → 1</li>
                    <li>Groceries → 2</li>
                    <li>Restaurants → 3</li>
                    <li>Travel → 4</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2c2:
        st.markdown(
            """
            <div class="card">
                <h4>Transaction Type Encoding</h4>
                <ul>
                    <li>ATM Withdrawal → 0</li>
                    <li>Bank Transfer → 1</li>
                    <li>Online → 2</li>
                    <li>POS → 3</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    
    df = load_preprocessed_data()
    if df.empty:
        st.warning("Preprocessed data not found.")
        return

    # --- DATASET SHAPE CARD ---
    st.markdown(
        f"""
        <div class="card">
            <p><b>Dataset shape:</b> {df.shape[0]} rows × {df.shape[1]} columns</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SHOW DATAFRAME ---
    st.dataframe(df.head(100), use_container_width=True)


def page_eda():
    st.markdown("<h2>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    df = load_preprocessed_data()
    if df.empty:
        st.warning("Data not available.")
        return

    choice = st.selectbox(
        "Select EDA Plot",
        [
            "Transaction Amount Distribution",
            "Account Balance Distribution",
            "Fraud Rate by Device Type",
            "Top Fraud Locations",
            "Correlation Heatmap"
        ]
    )

    if choice == "Transaction Amount Distribution":
        fig, ax = plt.subplots()
        sns.histplot(df["Transaction_Amount"], kde=True, bins=50, ax=ax)
        st.pyplot(fig)

    elif choice == "Account Balance Distribution":
        fig, ax = plt.subplots()
        sns.histplot(df["Account_Balance"], kde=True, bins=50, ax=ax)
        st.pyplot(fig)

    elif choice == "Fraud Rate by Device Type":
        fig, ax = plt.subplots()
        sns.barplot(x=df.groupby("Device_Type")["Fraud_Label"].mean().index,
                    y=df.groupby("Device_Type")["Fraud_Label"].mean().values,
                    ax=ax)
        st.pyplot(fig)

    elif choice == "Top Fraud Locations":
        fraud_locs = (
            df[df["Fraud_Label"] == 1]
            .groupby("Location")["Fraud_Label"]
            .count()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots()
        sns.barplot(y=fraud_locs.index, x=fraud_locs.values, ax=ax)
        st.pyplot(fig)

    elif choice == "Correlation Heatmap":

        # --- Drop these unwanted columns before correlation ---
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

        # Keep only numeric columns
        numeric_df = df2.select_dtypes(include=['int64', 'float64'])

        if numeric_df.shape[1] < 2:
            st.info("Not enough numeric columns for correlation heatmap.")
            return

        corr = numeric_df.corr()

        # --- FULL CORRELATION HEATMAP ---
        fig, ax = plt.subplots(figsize=(14,10))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        # --- TRIANGULAR CORRELATION HEATMAP ---
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig2, ax2 = plt.subplots(figsize=(14,10))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", mask=mask, ax=ax2)
        ax2.set_title("Triangular Correlation Heatmap")
        st.pyplot(fig2)

def page_ml_results():
    st.markdown("<h2>ML Model Results</h2>", unsafe_allow_html=True)

    df = load_results_df()
    if df.empty:
        st.warning("final_results.csv not found.")
        return

    st.dataframe(df)

def page_ml_plots():
    st.markdown("<h2>ML Diagnostic Plots</h2>", unsafe_allow_html=True)

    if not os.path.isdir(PLOTS_DIR):
        st.warning("Plot folder missing.")
        return

    plot_files = sorted([f for f in os.listdir(PLOTS_DIR) if f.lower().endswith(".png")])

    if not plot_files:
        st.info("No plot files available.")
        return

    models = sorted({f.split("_ROC")[0].split("_PR")[0].split("_CM")[0] for f in plot_files})
    selected = st.selectbox("Select Model", models)

    cols = st.columns(3)
    for name, col in zip(["ROC", "PR", "CM"], cols):
        file = os.path.join(PLOTS_DIR, f"{selected}_{name}.png")
        with col:
            if os.path.exists(file):
                st.image(file)
            else:
                st.write("Not available")

def page_realtime():
    st.markdown("<h2>Real-Time Monitoring</h2>", unsafe_allow_html=True)

    df = load_stream_outputs()
    if df.empty:
        st.info("No stream data yet.")
        return

    st.markdown("Latest Transactions")
    st.dataframe(df.tail(20).style.apply(highlight_fraud, axis=1), use_container_width=True)

    if "Fraud_Prediction" in df.columns:
        total = len(df)
        fraud = df["Fraud_Prediction"].sum()

        st.markdown("Live Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Transactions", total)
        with col2:
            st.metric("Fraudulent Transactions", fraud)

# ============================================
# PAGE ROUTING
# ============================================
page = st.session_state["active_page"]

if page == "Overview":
    page_overview()
elif page == "Preprocessed Data":
    page_preprocessed_data()
elif page == "EDA":
    page_eda()
elif page == "ML Model Results":
    page_ml_results()
elif page == "ML Plots":
    page_ml_plots()
elif page == "Real-Time Monitoring":
    page_realtime()
else:
    page_overview()
