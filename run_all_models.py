# run_all_models.py
# -------------------------------------------------------
# FULL ML PIPELINE RUNNER
# Generates:
# - final_results.csv
# - all ROC / PR / confusion matrix plots (in /plots)
# - best_fraud_model.pkl
# - fraud_scaler.pkl
# - feature_columns.json
# -------------------------------------------------------

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from data_loader import load_data, get_ml_dataset
from ml_models import run_full_pipeline, plot_roc, plot_pr, plot_cm

# -------------------------------------------------------
# CREATE SAVE DIRECTORIES
# -------------------------------------------------------
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------------------------------------------
# SAVE ALL PLOTS PER MODEL
# -------------------------------------------------------
def save_plots(y_test, probs_dict, preds_dict):
    print("\n[INFO] Saving model plots...")

    for model_name, prob in probs_dict.items():
        try:
            preds = preds_dict.get(model_name)
            if preds is None:
                # if preds not stored, derive from prob with 0.5 threshold
                preds = (prob >= 0.5).astype(int)

            # --- ROC ---
            try:
                plot_roc(y_test, prob, model_name, show=False)
                plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_ROC.png"), dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not save ROC for {model_name}: {e}")

            # --- Precision-Recall ---
            try:
                plot_pr(y_test, prob, model_name, show=False)
                plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_PR.png"), dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not save PR for {model_name}: {e}")

            # --- Confusion Matrix ---
            try:
                plot_cm(y_test, preds, model_name, show=False)
                plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_CM.png"), dpi=300, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not save CM for {model_name}: {e}")

        except Exception as e:
            print(f"[WARN] Skipping plots for {model_name}: {e}")

    print("[INFO] All plots saved in 'plots/' folder.")


# -------------------------------------------------------
# SELECT BEST MODEL USING ALL METRICS
# -------------------------------------------------------
def get_best_model(results_df, models_dict):
    """
    Select best supervised model using a composite score of:
    F1, Recall, Precision, ROC_AUC
    Only uses rows where Stage != 'Unsupervised'.
    """

    print("\n[INFO] Selecting BEST model using all metrics (F1, Recall, Precision, ROC_AUC)...")

    supervised_df = results_df[results_df["Stage"] != "Unsupervised"].copy()

    if supervised_df.empty:
        raise ValueError("No supervised models found in results_df to choose best model from.")

    # Fill NaNs with 0 for safety
    supervised_df = supervised_df.fillna(0.0)

    # Composite score (you can explain this in viva)
    # More weight on F1 and Recall, then Precision, then ROC_AUC
    supervised_df["Overall_Score"] = (
        0.4 * supervised_df["F1 Score"] +
        0.3 * supervised_df["Recall"] +
        0.2 * supervised_df["Precision"] +
        0.1 * supervised_df["ROC_AUC"]
    )

    best_row = supervised_df.loc[supervised_df["Overall_Score"].idxmax()]
    best_model_name = best_row["Model"]

    print("\n[INFO] Best model row:")
    print(best_row)

    if best_model_name not in models_dict:
        raise KeyError(f"Best model '{best_model_name}' not found in models_dict keys: {list(models_dict.keys())}")

    best_model = models_dict[best_model_name]

    print(f"\n🔥 BEST MODEL SELECTED: {best_model_name}")
    print(f"⭐ F1 Score: {best_row['F1 Score']:.4f}")
    print(f"⭐ Recall:   {best_row['Recall']:.4f}")
    print(f"⭐ Precision:{best_row['Precision']:.4f}")
    print(f"⭐ ROC_AUC:  {best_row['ROC_AUC']:.4f}")
    print(f"⭐ Overall Score: {best_row['Overall_Score']:.4f}")

    return best_model_name, best_model


# -------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------
def main():

    print("\n===============================")
    print(" RUNNING FULL FRAUD ML PIPELINE")
    print("===============================\n")

    # --------------------------------------
    # LOAD RAW DATA
    # --------------------------------------
    df_raw = load_data()
    df_ml = get_ml_dataset(df_raw)

    # --------------------------------------
    # RUN FULL PIPELINE
    # --------------------------------------
    (
        results_df,
        all_supervised_models,
        probs_dict,
        preds_dict,
        scaler,
        X_train_s,
        X_test_s,
        y_train,
        y_test
    ) = run_full_pipeline(df_ml)

    # Save final dataframe
    results_df.to_csv("final_results.csv", index=False)
    print("\n[INFO] Final ML results saved → final_results.csv")

    # --------------------------------------
    # SAVE ALL PLOTS
    # --------------------------------------
    save_plots(y_test, probs_dict, preds_dict)

    # --------------------------------------
    # CHOOSE BEST MODEL
    # --------------------------------------
    best_model_name, best_model = get_best_model(results_df, all_supervised_models)

    # --------------------------------------
    # SAVE BEST MODEL + SCALER + FEATURES
    # --------------------------------------
    print("\n[INFO] Saving best model and scaler...")

    joblib.dump(best_model, "best_fraud_model.pkl")
    joblib.dump(scaler, "fraud_scaler.pkl")

    feature_list = df_ml.drop(columns=["Fraud_Label"]).columns.tolist()
    with open("feature_columns.json", "w") as f:
        json.dump(feature_list, f, indent=4)

    print("[INFO] best_fraud_model.pkl saved")
    print("[INFO] fraud_scaler.pkl saved")
    print("[INFO] feature_columns.json saved")

    print("\n🎉 ML Training Completed Successfully!")
    print("All outputs ready for STREAMLIT + REAL-TIME FRAUD DETECTION.")


if __name__ == "__main__":
    main()
