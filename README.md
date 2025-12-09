# 💳 Real-Time Financial Fraud Detection System  
*A complete end-to-end machine learning pipeline with real-time streaming, monitoring dashboard, and automated email alerts.*

---

## 🚀 Project Overview  

This project implements a complete **Real-Time Financial Fraud Detection System** designed to identify suspicious transactions the moment they occur. It includes:

- Full **data preprocessing pipeline** (cleaning, encoding, scaling, feature engineering)
- Handling class imbalance using **SMOTE / SMOTE-ENN**
- Training multiple **ML models**  
  *(Logistic Regression, Random Forest, XGBoost, LightGBM, etc.)*
- Model evaluation using **F1-Score, Precision, Recall, ROC-AUC**
- Exporting the best-performing model for deployment
- A **Python-based streaming engine** that simulates real-world banking transactions
- A **real-time Streamlit dashboard** to visualize fraud alerts & predictions
- **Automatic email alerts** for high-risk transactions

> ⚠️ Note:  
> Model `.pkl` files are not included in this repository because of size limits.  
> They will automatically be generated when you run the ML training script.

---
📂 project_root
project/
│
├── data/
│ └── preprocessed_synthetic_fraud_data.csv
│
├── fraud_streaming/
│ ├── auto_generate_stream.py
│ ├── python_fraud_stream.py
│ └── stream_output/
└── stream_input/
│
├── plots/
│
├── pages/
│ └── 1_Real_Time_Monitoring.py
│
├── app.py
├── run_all_models.py
├── README.md
└── (model & scaler files generated after training)


---

## 🔧 Installation & Setup  
### **Create a virtual environment and Activate it**
### ** Install Dependencies : **pip install -r requirements.txt**

---
### 🧠 Train Models (Auto-generate .pkl files)

Run this script first: python run_all_models.py

This will:

✔ Train all ML models
✔ Evaluate and compare them
✔ Generate plots (ROC, PR, Confusion Matrix)
✔ Export the best model and scaler into root folder

## Start Real-Time Fraud Detection
1️⃣ Start the streaming prediction engine : python fraud_streaming/python_fraud_stream.py

2️⃣ Start the simulated live transaction generator (This creates 10 csv files automatically for detect fraud data and send it as notification alert via mail) : python fraud_streaming/auto_generate_stream.py

3️⃣ Run the dashboard : streamlit run app.py

## 📊 Streamlit Dashboard Features
## 1. Overview

    Project introduction
    Architecture summary
    Workflow explanation


## 2. Preprocessed Data

    First 100 rows preview
    Encoding explanation (categorical → numerical)
    Dataset summary

## 3. EDA (Exploratory Data Analysis)

    Transaction amount distribution
    Account balance distribution
    Fraud rate by device
    Fraud locations
    Correlation heatmap

## 4. ML Model Results

    Metrics comparison table
    Best model highlight

## 5. ML Plots

    ROC curves
    Precision–Recall curves
    Confusion matrices

## 6. Real-Time Monitoring

    Auto-updating table of latest transactions
    Fraud row highlighting
    Fraud analytics (metrics + graphs)
    Model performance on streaming data
    Alerts summary

## ✉️ Email Alerts

    Whenever a transaction is predicted as Fraud_Prediction = 1, the system:
        Highlights it in the Streamlit UI
        Sends an automated fraud alert email
        Configured in alerts.py.

## 👥 Team Members

    - Swarnathara Ramesh
    - Akash Kar Choudary
    - Rajnandani Godage

## ⭐ Future Enhancements

    - Deploy model as a REST API
    - Integrate message queues (Kafka / RabbitMQ)
    - Add deep learning models (LSTM for sequence fraud detection)
    - Deploy dashboard on cloud (AWS/GCP/Render)

