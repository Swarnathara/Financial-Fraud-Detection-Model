import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------------
# PATHS
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STREAM_DIR = os.path.join(BASE_DIR, "fraud_streaming", "stream_output")

# -----------------------------------
# HELPERS
# -----------------------------------
def load_stream():
    if not os.path.isdir(STREAM_DIR):
        return pd.DataFrame()

    files = [
        os.path.join(STREAM_DIR, f)
        for f in os.listdir(STREAM_DIR)
        if f.lower().endswith(".csv")
    ]

    dfs = []
    for fp in sorted(files):
        try:
            df = pd.read_csv(fp)
            df["__file"] = os.path.basename(fp)
            dfs.append(df)
        except:
            pass

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def highlight_fraud(row):
    return ["background-color: #4a0c0c" if row.get("Fraud_Prediction", 0) == 1 else ""] * len(row)


# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Real-Time Monitoring",
    layout="wide"
)

st.title("Real-Time Fraud Monitoring")

df = load_stream()

if df.empty:
    st.warning("No streaming data yet. Start the streaming engine.")
    st.stop()

# -----------------------------------
# SIDEBAR MENU
# -----------------------------------
SECTIONS = [
    "Live Transactions",
    "Fraud Analytics",
    "Model Performance",
    "Alerts Summary"
]

if "section" not in st.session_state:
    st.session_state["section"] = "Live Transactions"

with st.sidebar:
    st.header("Menu")
    for sec in SECTIONS:
        if st.button(sec, use_container_width=True):
            st.session_state["section"] = sec
            st.rerun()

section = st.session_state["section"]

# -----------------------------------
# SECTION 1 — LIVE TRANSACTIONS
# -----------------------------------
if section == "Live Transactions":
    st.subheader("Latest Transactions")

    N = st.slider("Show latest N", 10, 200, 30)
    st.dataframe(df.tail(N).style.apply(highlight_fraud, axis=1), use_container_width=True)


# -----------------------------------
# SECTION 2 — FRAUD ANALYTICS
# -----------------------------------
elif section == "Fraud Analytics":
    st.subheader("Fraud Analytics Overview")

    total = len(df)
    fraud = df["Fraud_Prediction"].sum()
    rate = fraud / total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Fraudulent", fraud)
    col3.metric("Fraud Rate (%)", f"{rate:.2f}")

    st.markdown("### GRAPHICAL ANALYSIS")

    g1, g2 = st.columns(2)

    # ---- Graph 1: Fraud vs Normal ----
    with g1:
        st.markdown("#### Fraud vs Normal")
        fig1, ax1 = plt.subplots()
        ax1.bar(["Normal", "Fraud"], [total - fraud, fraud], color=["#34d399", "#ef4444"])
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    # ---- Graph 2: Fraud by Location or Merchant ----
    with g2:
        if "Location" in df.columns:
            st.markdown("#### Fraud by Location (Top 10)")
            f_loc = (
                df[df["Fraud_Prediction"] == 1]
                .groupby("Location")["Fraud_Prediction"]
                .count()
                .sort_values(ascending=False)
                .head(10)
            )
            fig2, ax2 = plt.subplots()
            sns.barplot(y=f_loc.index, x=f_loc.values, ax=ax2)
            ax2.set_xlabel("Fraud Count")
            st.pyplot(fig2)

        elif "Merchant_Category" in df.columns:
            st.markdown("#### Fraud by Merchant Category (Top 10)")
            f_merch = (
                df[df["Fraud_Prediction"] == 1]
                .groupby("Merchant_Category")["Fraud_Prediction"]
                .count()
                .sort_values(ascending=False)
                .head(10)
            )
            fig3, ax3 = plt.subplots()
            sns.barplot(y=f_merch.index, x=f_merch.values, ax=ax3)
            ax3.set_xlabel("Fraud Count")
            st.pyplot(fig3)
# -----------------------------------
# SECTION 3 — MODEL PERFORMANCE
# -----------------------------------
elif section == "Model Performance":
    st.subheader("Model Performance (Live Stream)")

    if "Fraud_Label" not in df.columns:
        st.info("True labels missing. Cannot compute performance.")
    else:
        y_true = df["Fraud_Label"].astype(int)
        y_pred = df["Fraud_Prediction"].astype(int)

        # ---- FIRST: Classification Report Table (Static) ----
        st.markdown("### Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        st.markdown("---")

        # ---- SECOND: Confusion Matrix (Below the table) ----
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# -----------------------------------
# SECTION 4 — ALERTS SUMMARY
# -----------------------------------
elif section == "Alerts Summary":
    st.subheader("Alerts Summary (Fraud Only)")

    fraud_rows = df[df["Fraud_Prediction"] == 1]

    if fraud_rows.empty:
        st.info("No alerts yet.")
    else:
        st.dataframe(fraud_rows.tail(20).style.apply(highlight_fraud, axis=1), use_container_width=True)
