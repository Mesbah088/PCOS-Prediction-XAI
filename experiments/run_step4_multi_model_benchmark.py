import os
import sys
import re
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

from src.preprocessing import load_and_preprocess_data
from src.models import (
    get_comprehensive_base_models,
    get_champion_soft_voting_ensemble,
    get_champion_stacking_ensemble,
    get_calibrated_champion,
    evaluate_clinical_model,
    compute_cross_val_metrics,
    perform_wilcoxon_significance
)

# Canonical 16 invariant biomarkers from Step 3 Tri-HFS
STEP3_SELECTED_16 = [
    "PCOM_Severe_Flag",
    "PCOM_Rotterdam_Flag",
    "Cycle_R_I",
    "Skin_Metabolic_Score",
    "Follicle_No._R",
    "Weight_gain_Y_N",
    "Follicles_Per_Year",
    "hair_growth_Y_N",
    "Total_Follicles",
    "Follicle_Geometric_Mean",
    "Hirsutism_Acne_Score",
    "Metabolic_Symptom_Score",
    "Fast_food_Y_N",
    "AMH_Total_Follicles_Interaction",
    "Hair_loss_Y_N",
    "Weight_Kg"
]

def sanitize_name(col: str) -> str:
    return re.sub(r'[:,\(\)\[\]\{\}\"\/\\ ]+', '_', col).strip('_')

def run_step4_benchmark():
    os.makedirs("outputs", exist_ok=True)
    print("=" * 85)
    print("  STEP 4: MULTI-CLASSIFIER BENCHMARK & CHAMPION SUPER-ENSEMBLE TRAINING")
    print("  Evaluating 12 Base Models + 2 Meta-Ensembles + Calibrated Champion")
    print("  Cohort: 541 Patients (432 Train -> Resampled via SMOTE-ENN; 109 Held-Out Test)")
    print("=" * 85)

    # 1. Zero-Leakage Preprocessing & Ingestion
    data_path = os.path.join("data", "PCOS_data.csv")
    X_train_raw, X_test_raw, y_train, y_test, _, _ = load_and_preprocess_data(
        file_path=data_path,
        apply_feature_engineering=True,
        resampling_strategy="smote_enn",
        test_size=0.20,
        random_state=42
    )

    # Sanitize feature column names
    col_map = {c: sanitize_name(c) for c in X_train_raw.columns}
    X_tr_all = X_train_raw.rename(columns=col_map)
    X_te_all = X_test_raw.rename(columns=col_map)

    # Subset strictly to the 16 invariant biomarkers
    sanitized_16 = [c for c in STEP3_SELECTED_16 if c in X_tr_all.columns]
    X_train = X_tr_all[sanitized_16]
    X_test = X_te_all[sanitized_16]

    print(f"\n[Feature Space] Locked to the 16 Tri-HFS Invariant Biomarkers:")
    for idx, f in enumerate(sanitized_16, 1):
        print(f"  {idx:2d}. {f}")

    # 2. Build Model Suite
    base_models = get_comprehensive_base_models(random_state=42)
    soft_ensemble = get_champion_soft_voting_ensemble(random_state=42)
    stack_ensemble = get_champion_stacking_ensemble(random_state=42)

    # Assemble all candidates
    models_to_evaluate = {}
    # Top ensembles first
    models_to_evaluate["🥇 Champion Super-Ensemble (Soft-Voting)"] = soft_ensemble
    models_to_evaluate["🥈 Champion Stacking Meta-Learner"] = stack_ensemble
    # Base models
    for name, m in base_models.items():
        models_to_evaluate[name] = m

    # 3. Benchmark Execution
    results = []
    roc_data = {}
    fitted_models = {}
    cv_fold_f1_cache = {}

    print("\n" + "=" * 85, flush=True)
    print(" EXECUTING BENCHMARK ACROSS ALL MODELS (5-Fold CV & Held-Out Test Evaluation)", flush=True)
    print("=" * 85, flush=True)

    # First evaluate Champion to use as baseline for Wilcoxon tests
    champ_key = "🥇 Champion Super-Ensemble (Soft-Voting)"
    print(f"-> Training Champion Reference: {champ_key}...", flush=True)
    champ_eval = evaluate_clinical_model(soft_ensemble, X_train, y_train, X_test, y_test)
    champ_cv = compute_cross_val_metrics(soft_ensemble, X_train, y_train, n_splits=5, random_state=42)
    champ_f1_folds = champ_cv["fold_f1_scores"]

    for name, model in models_to_evaluate.items():
        print(f"-> Evaluating: {name}...", flush=True)
        eval_metrics = evaluate_clinical_model(model, X_train, y_train, X_test, y_test)
        cv_metrics = compute_cross_val_metrics(model, X_train, y_train, n_splits=5, random_state=42)
        fitted_models[name] = eval_metrics["model"]
        roc_data[name] = (eval_metrics["y_proba"], eval_metrics["ROC-AUC"])

        # Wilcoxon test vs Champion
        p_val, is_sig = perform_wilcoxon_significance(champ_f1_folds, cv_metrics["fold_f1_scores"])
        sig_str = "Ref" if name == champ_key else (f"{p_val:.4f}*" if is_sig else f"{p_val:.4f}")

        results.append({
            "Model Name": name,
            "Accuracy (%)": eval_metrics["Accuracy (%)"],
            "Recall / Sensitivity (%)": eval_metrics["Recall (%)"],
            "Specificity (%)": eval_metrics["Specificity (%)"],
            "Precision (%)": eval_metrics["Precision (%)"],
            "F1-Score (%)": eval_metrics["F1-Score (%)"],
            "ROC-AUC": eval_metrics["ROC-AUC"],
            "Brier Score": eval_metrics["Brier Score"],
            "FN (Missed)": eval_metrics["FN"],
            "FP (Alarms)": eval_metrics["FP"],
            "5-Fold CV F1 (%)": cv_metrics["CV_F1_Str"],
            "Wilcoxon p-val": sig_str
        })

    # 4. Calibrated Champion (Platt Scaling)
    print("\n-> Fitting Post-Hoc Calibrated Champion (Platt Scaling Sigmoid)...", flush=True)
    calibrated_champion = get_calibrated_champion(soft_ensemble, method='sigmoid')
    cal_eval = evaluate_clinical_model(calibrated_champion, X_train, y_train, X_test, y_test)
    cal_cv = compute_cross_val_metrics(calibrated_champion, X_train, y_train, n_splits=5, random_state=42)
    fitted_models["⭐ Calibrated Champion (Platt Scaling)"] = cal_eval["model"]
    roc_data["⭐ Calibrated Champion (Platt Scaling)"] = (cal_eval["y_proba"], cal_eval["ROC-AUC"])

    results.append({
        "Model Name": "⭐ Calibrated Champion (Platt Scaling)",
        "Accuracy (%)": cal_eval["Accuracy (%)"],
        "Recall / Sensitivity (%)": cal_eval["Recall (%)"],
        "Specificity (%)": cal_eval["Specificity (%)"],
        "Precision (%)": cal_eval["Precision (%)"],
        "F1-Score (%)": cal_eval["F1-Score (%)"],
        "ROC-AUC": cal_eval["ROC-AUC"],
        "Brier Score": cal_eval["Brier Score"],
        "FN (Missed)": cal_eval["FN"],
        "FP (Alarms)": cal_eval["FP"],
        "5-Fold CV F1 (%)": cal_cv["CV_F1_Str"],
        "Wilcoxon p-val": "Calibrated"
    })

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 115)
    print(" [BENCHMARK RESULTS] MULTI-CLASSIFIER COMPARISON ON LOCKED 109 HELD-OUT TEST PATIENTS")
    print("=" * 115)
    print(df_results.to_string(index=False))

    # Save CSV
    csv_path = os.path.join("outputs", "multi_model_benchmark_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Saved CSV] Benchmark table saved to: {csv_path}")

    # 5. Serialize Champion Model
    model_save_path = os.path.join("outputs", "champion_ensemble_model.joblib")
    joblib.dump({
        "model": calibrated_champion,
        "feature_names": sanitized_16,
        "scaler": None, # Features are already pre-scaled
        "metrics": cal_eval
    }, model_save_path)
    print(f"[Saved Model] Serialized Champion Model saved to: {model_save_path}")

    # 6. Generate Publication Charts
    # Chart 1: Grouped Performance Comparison
    plt.figure(figsize=(14, 7), dpi=300)
    top_models_for_plot = df_results[~df_results["Model Name"].str.contains("Calibrated")].copy()
    plot_df = pd.melt(
        top_models_for_plot,
        id_vars=["Model Name"],
        value_vars=["Accuracy (%)", "Recall / Sensitivity (%)", "Specificity (%)", "F1-Score (%)"],
        var_name="Clinical Metric",
        value_name="Score (%)"
    )
    ax = sns.barplot(data=plot_df, x="Model Name", y="Score (%)", hue="Clinical Metric", palette="viridis")
    plt.title("Step 4: Multi-Model Clinical Benchmark Comparison (16 Invariant Rotterdam Biomarkers)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Classifier Architecture", fontsize=11)
    plt.ylabel("Score (%)", fontsize=11)
    plt.ylim(65, 102)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart1_path = os.path.join("outputs", "multi_model_performance_chart.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Performance chart saved to: {chart1_path}")

    # Chart 2: ROC-AUC Curves
    plt.figure(figsize=(9, 7), dpi=300)
    selected_roc_models = [
        "🥇 Champion Super-Ensemble (Soft-Voting)",
        "LightGBM",
        "CatBoost",
        "XGBoost",
        "Random Forest",
        "MLP (Neural Net)",
        "SVM (RBF Kernel)"
    ]
    colors_list = ['#DC2626', '#2563EB', '#059669', '#D97706', '#7C3AED', '#DB2777', '#4B5563']
    for idx, m_name in enumerate(selected_roc_models):
        if m_name in roc_data:
            y_prob, auc_score = roc_data[m_name]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_score:.4f})", color=colors_list[idx % len(colors_list)], lw=2.2 if idx==0 else 1.5)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.5000)")
    plt.title("Receiver Operating Characteristic (ROC) Curves Across Diagnostic Architectures", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    plt.legend(loc="lower right", fontsize=8.5, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    chart2_path = os.path.join("outputs", "multi_model_roc_curves.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] ROC curves saved to: {chart2_path}")

    # Chart 3: Calibration Curve (Reliability Diagram)
    plt.figure(figsize=(8, 6), dpi=300)
    # Uncalibrated Champion
    uncal_prob = champ_eval["y_proba"]
    fop_uncal, mpv_uncal = calibration_curve(y_test, uncal_prob, n_bins=6)
    # Calibrated Champion
    cal_prob = cal_eval["y_proba"]
    fop_cal, mpv_cal = calibration_curve(y_test, cal_prob, n_bins=6)

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", lw=1.5)
    plt.plot(mpv_uncal, fop_uncal, "s-", color="#DC2626", label=f"Uncalibrated Ensemble (Brier = {champ_eval['Brier Score']:.4f})", lw=2)
    plt.plot(mpv_cal, fop_cal, "o-", color="#059669", label=f"Platt-Calibrated Champion (Brier = {cal_eval['Brier Score']:.4f})", lw=2.2)

    plt.title("Clinical Probability Calibration Curves (Reliability Diagram)", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Mean Predicted Disease Probability", fontsize=10)
    plt.ylabel("Observed Fraction of True PCOS Patients", fontsize=10)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    chart3_path = os.path.join("outputs", "champion_calibration_curves.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Calibration curves saved to: {chart3_path}")

    # Chart 4: Champion Confusion Matrix
    plt.figure(figsize=(6.5, 5.5), dpi=300)
    cm = confusion_matrix(y_test, cal_eval["y_pred"])
    labels = ["Control (Non-PCOS)", "PCOS Positive"]
    annotations = [
        [f"TN = {cm[0, 0]}\n({cm[0, 0]/73*100:.1f}%)", f"FP = {cm[0, 1]}\n({cm[0, 1]/73*100:.1f}%)"],
        [f"FN = {cm[1, 0]}\n({cm[1, 0]/36*100:.1f}%)", f"TP = {cm[1, 1]}\n({cm[1, 1]/36*100:.1f}%)"]
    ]
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 11, "fontweight": "bold"})
    plt.title(f"Champion Super-Ensemble Confusion Matrix (N = 109 Test Patients)\nSensitivity: {cal_eval['Recall (%)']:.2f}% | Specificity: {cal_eval['Specificity (%)']:.2f}%", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Predicted Clinical Category", fontsize=10)
    plt.ylabel("Ground Truth Medical Diagnosis", fontsize=10)
    plt.tight_layout()
    chart4_path = os.path.join("outputs", "champion_confusion_matrix.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Confusion matrix saved to: {chart4_path}")

    print("\n" + "=" * 85)
    print(" Step 4 Multi-Classifier Benchmark Successfully Completed!")
    print("=" * 85)
    return df_results

if __name__ == "__main__":
    run_step4_benchmark()
