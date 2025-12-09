# ml_models.py
"""
Machine Learning pipeline module.
Supports:
 - Supervised: LogisticRegression, DecisionTree, RandomForest, XGBoost, LightGBM
 - Class-weight variants
 - SMOTEENN variants
 - Unsupervised: IsolationForest, OneClassSVM, PCA reconstruction, Autoencoder (MLPRegressor)
 - Plots: ROC, Precision-Recall, Confusion Matrix
 - Returns final combined results DataFrame
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")


# ---------------------
# Helper metric function
# ---------------------
def compute_metrics(y_true, y_pred, y_score=None):
    """Return dict of key metrics. y_score can be None."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = None
    if y_score is not None:
        try:
            roc = roc_auc_score(y_true, y_score)
        except:
            roc = None
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC_AUC": roc if roc is not None else np.nan
    }


# ---------------------
# Train / evaluate supervised baseline models
# ---------------------
def train_baseline_models(X_train, y_train):
    """Train baseline supervised models and return dict{name: model}"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    }

    for name, m in models.items():
        print(f"[TRAIN] {name}")
        m.fit(X_train, y_train)

    return models


# ---------------------
# Train class-weight models (where applicable)
# ---------------------
def train_classweight_models(X_train, y_train):
    models = {
        "Logistic Regression (CW)": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Random Forest (CW)": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "XGBoost (CW)": XGBClassifier(n_estimators=300, learning_rate=0.05,
                                      scale_pos_weight=( (len(y_train)-y_train.sum())/y_train.sum() ),
                                      use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    for name, m in models.items():
        print(f"[TRAIN] {name}")
        m.fit(X_train, y_train)
    return models


# ---------------------
# Train SMOTE / SMOTEENN versions
# ---------------------
def train_smote_models(X_train, y_train):
    # SMOTEENN combine method
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    print("[INFO] SMOTEENN applied. Resampled shape:", X_res.shape)

    models = {
        "Random Forest (SMOTEENN)": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost (SMOTEENN)": XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM (SMOTEENN)": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    }

    for name, m in models.items():
        print(f"[TRAIN] {name}")
        m.fit(X_res, y_res)

    return models


# ---------------------
# Unsupervised methods
# ---------------------
def run_unsupervised(X_train, X_test, y_test, contamination=0.02):
    """
    Run IsolationForest, OneClassSVM, PCA reconstruction, Autoencoder (MLPRegressor).
    Returns results list (dicts) and score/prediction dicts for plotting.
    """
    results = []
    score_dict = {}
    pred_dict = {}

    # Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_train)
    iso_pred = np.where(iso.predict(X_test) == -1, 1, 0)
    iso_score = -iso.decision_function(X_test)  # higher => more anomalous
    results.append({"Model": "IsolationForest", **compute_metrics(y_test, iso_pred, iso_score), "Stage": "Unsupervised"})
    score_dict["IsolationForest"] = iso_score
    pred_dict["IsolationForest"] = iso_pred

    # One-Class SVM
    oc = OneClassSVM(kernel='rbf', gamma='auto', nu=contamination)
    oc.fit(X_train)
    oc_pred = np.where(oc.predict(X_test) == -1, 1, 0)
    try:
        oc_score = -oc.decision_function(X_test)
    except:
        oc_score = oc_pred  # fallback
    results.append({"Model": "OneClassSVM", **compute_metrics(y_test, oc_pred, oc_score), "Stage": "Unsupervised"})
    score_dict["OneClassSVM"] = oc_score
    pred_dict["OneClassSVM"] = oc_pred

    # PCA reconstruction error
    pca = PCA(n_components=min(0.95, 0.99)) if isinstance(0.95, float) else PCA(n_components=min(X_train.shape[1], 10))
    # We'll try to set n_components to explain 95% variance; fallback to fixed components if sklearn complains.
    try:
        pca = PCA(n_components=0.95, random_state=42)
        pca.fit(X_train)
        X_test_proj = pca.transform(X_test)
        X_test_recon = pca.inverse_transform(X_test_proj)
    except Exception:
        # fallback: fixed n comps
        pca = PCA(n_components=min(10, X_train.shape[1]))
        pca.fit(X_train)
        X_test_recon = pca.inverse_transform(pca.transform(X_test))

    pca_error = np.mean((X_test - X_test_recon) ** 2, axis=1)
    pca_thresh = np.percentile(np.mean((X_train - pca.inverse_transform(pca.transform(X_train)))**2, axis=1), 95)
    pca_pred = (pca_error > pca_thresh).astype(int)
    results.append({"Model": "PCA_Reconstruction", **compute_metrics(y_test, pca_pred, pca_error), "Stage": "Unsupervised"})
    score_dict["PCA_Reconstruction"] = pca_error
    pred_dict["PCA_Reconstruction"] = pca_pred

    # Autoencoder-like using MLPRegressor
    ae = MLPRegressor(hidden_layer_sizes=(64, 32, 64), max_iter=200, random_state=42)
    # Train to reconstruct
    try:
        ae.fit(X_train, X_train)
        X_test_rec = ae.predict(X_test)
        ae_error = np.mean((X_test - X_test_rec) ** 2, axis=1)
        ae_thresh = np.percentile(np.mean((X_train - ae.predict(X_train))**2, axis=1), 95)
        ae_pred = (ae_error > ae_thresh).astype(int)
        results.append({"Model": "Autoencoder_MLP", **compute_metrics(y_test, ae_pred, ae_error), "Stage": "Unsupervised"})
        score_dict["Autoencoder_MLP"] = ae_error
        pred_dict["Autoencoder_MLP"] = ae_pred
    except Exception as e:
        print("[WARN] Autoencoder training failed:", e)

    return results, score_dict, pred_dict


# ---------------------
# Evaluate and produce plots
# ---------------------
def plot_roc(y_true, score, name, show=True):
    fpr, tpr, _ = roc_curve(y_true, score)
    auc_val = roc_auc_score(y_true, score)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.4f}")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.title(f"ROC — {name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def plot_pr(y_true, score, name, show=True):
    prec, rec, _ = precision_recall_curve(y_true, score)
    plt.figure(figsize=(6,4))
    plt.plot(rec, prec)
    plt.title(f"Precision-Recall — {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    if show:
        plt.show()


def plot_cm(y_true, y_pred, name, show=True):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if show:
        plt.show()


# ---------------------
# Full pipeline runner
# ---------------------
def run_full_pipeline(df_ml, test_size=0.2, random_state=42, save_results_path=None):
    """
    df_ml: ML-ready numeric dataframe including 'Fraud_Label' as target.
    Returns:
      - final_results_df: pandas DataFrame with metrics for all models/stages
      - models_dict: dictionary of trained baseline models (not sampling/classweight models)
      - probs_dict: dictionary mapping model_name -> probability scores (for plotting)
      - preds_dict: mapping model_name -> predicted labels
    """

    if 'Fraud_Label' not in df_ml.columns:
        raise ValueError("df_ml must contain 'Fraud_Label' target column")

    # 1. split & scale
    X = df_ml.drop(columns=['Fraud_Label'])
    y = df_ml['Fraud_Label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 2. Baseline supervised
    baseline_models = train_baseline_models(X_train_s, y_train)

    # Evaluate baseline models
    final_results = []
    probs_all = {}
    preds_all = {}

    for name, model in baseline_models.items():
        # probabilities / scores
        try:
            prob = model.predict_proba(X_test_s)[:, 1]
        except:
            score_raw = model.decision_function(X_test_s)
            prob = (score_raw - score_raw.min()) / (score_raw.max() - score_raw.min() + 1e-9)
        pred = (prob >= 0.5).astype(int)
        probs_all[name] = prob
        preds_all[name] = pred
        m = compute_metrics(y_test, pred, prob)
        m.update({"Model": name, "Stage": "Baseline"})
        final_results.append(m)

    # 3. Class-weight models
    cw_models = train_classweight_models(X_train_s, y_train)
    for name, model in cw_models.items():
        try:
            prob = model.predict_proba(X_test_s)[:,1]
        except:
            score_raw = model.decision_function(X_test_s)
            prob = (score_raw - score_raw.min()) / (score_raw.max() - score_raw.min() + 1e-9)
        pred = (prob >= 0.5).astype(int)
        probs_all[name] = prob
        preds_all[name] = pred
        m = compute_metrics(y_test, pred, prob)
        m.update({"Model": name, "Stage": "Class Weight"})
        final_results.append(m)

    # 4. SMOTEENN models
    smote_models = train_smote_models(X_train_s, y_train)
    for name, model in smote_models.items():
        try:
            prob = model.predict_proba(X_test_s)[:,1]
        except:
            score_raw = model.decision_function(X_test_s)
            prob = (score_raw - score_raw.min()) / (score_raw.max() - score_raw.min() + 1e-9)
        pred = (prob >= 0.5).astype(int)
        probs_all[name] = prob
        preds_all[name] = pred
        m = compute_metrics(y_test, pred, prob)
        m.update({"Model": name, "Stage": "SMOTEENN"})
        final_results.append(m)

        # 5. Unsupervised methods
    unsup_results, unsup_scores, unsup_preds = run_unsupervised(X_train_s, X_test_s, y_test)
    for r in unsup_results:
        final_results.append(r)
        name = r['Model']
        if name in unsup_scores:
            probs_all[name] = unsup_scores[name]
        if name in unsup_preds:
            preds_all[name] = unsup_preds[name]

    # ---- Build a dict of ALL supervised models (baseline + CW + SMOTEENN) ----
    all_supervised_models = {}
    all_supervised_models.update(baseline_models)
    all_supervised_models.update(cw_models)
    all_supervised_models.update(smote_models)

    # ---- Build final results dataframe ----
    final_df = pd.DataFrame(final_results)

    for c in ["Model", "Stage", "Accuracy", "Precision", "Recall", "F1 Score", "ROC_AUC"]:
        if c not in final_df.columns:
            final_df[c] = np.nan

    final_df = final_df[["Model", "Stage", "Accuracy", "Precision", "Recall", "F1 Score", "ROC_AUC"]]

    if save_results_path:
        final_df.to_csv(save_results_path, index=False)
        print("[INFO] Saved final results to", save_results_path)

    # NOTE: return all_supervised_models instead of baseline_models
    return final_df, all_supervised_models, probs_all, preds_all, scaler, X_train_s, X_test_s, y_train, y_test


# ---------------------
# Utility: plot all model diagnostics
# ---------------------
def plot_all_diagnostics(y_test, probs_dict, preds_dict, models_list=None):
    """
    Given y_test and dicts of probs & preds, plot ROC, PR, CM for each model.
    models_list optionally restricts to subset of keys.
    """
    keys = list(probs_dict.keys())
    if models_list:
        keys = [k for k in keys if k in models_list]

    for k in keys:
        prob = probs_dict[k]
        pred = preds_dict.get(k, (prob >= 0.5).astype(int))
        print(f"\n[DIAG] {k}")
        plot_roc(y_test, prob, k)
        plot_pr(y_test, prob, k)
        plot_cm(y_test, pred, k)


