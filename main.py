import os
import sys
import glob
import pandas as pd
import numpy as np

from src.preprocessing import load_and_preprocess_data
from src.feature_selection import two_stage_feature_selection
from src.models import (
    get_base_models,
    get_champion_ensemble,
    evaluate_model,
    perform_10fold_cv
)
from src.xai import generate_shap_explanations, generate_lime_explanation

def create_synthetic_demo_data(file_path: str):
    """Creates synthetic placeholder dataset matching the PCOS schema if no data is found."""
    np.random.seed(42)
    n_samples = 541
    data = {
        "Sl. No": range(1, n_samples + 1),
        "Patient File No.": [f"PF_{i}" for i in range(1, n_samples + 1)],
        "PCOS (Y/N)": np.random.choice([0, 1], size=n_samples, p=[0.673, 0.327]),
        "Age (years)": np.random.randint(18, 45, size=n_samples),
        "Weight (Kg)": np.random.normal(60, 12, size=n_samples),
        "Height(Cm)": np.random.normal(160, 8, size=n_samples),
        "BMI": np.random.normal(24, 4, size=n_samples),
        "Blood Group": np.random.choice([11, 12, 13, 14, 15], size=n_samples),
        "Pulse rate(bpm)": np.random.randint(60, 100, size=n_samples),
        "RR (breaths/min)": np.random.randint(12, 25, size=n_samples),
        "Hb(g/dl)": np.random.normal(12.5, 1.5, size=n_samples),
        "Cycle(R/I)": np.random.choice([2, 4, 5], size=n_samples),
        "Cycle length(days)": np.random.randint(2, 12, size=n_samples),
        "Marraige Status (Yrs)": np.random.randint(0, 20, size=n_samples),
        "Pregnant(Y/N)": np.random.choice([0, 1], size=n_samples),
        "No. of aborptions": np.random.randint(0, 4, size=n_samples),
        "FSH(mIU/mL)": np.random.exponential(5, size=n_samples),
        "LH(mIU/mL)": np.random.exponential(4, size=n_samples),
        "FSH/LH": np.random.normal(1.5, 0.5, size=n_samples),
        "Hip(inch)": np.random.normal(38, 4, size=n_samples),
        "Waist(inch)": np.random.normal(32, 4, size=n_samples),
        "Waist:Hip Ratio": np.random.normal(0.85, 0.05, size=n_samples),
        "TSH (mIU/L)": np.random.normal(2.5, 1.0, size=n_samples),
        "AMH(ng/mL)": np.random.exponential(4.5, size=n_samples),
        "PRL(ng/mL)": np.random.normal(20, 5, size=n_samples),
        "Vit D3 (ng/mL)": np.random.normal(30, 10, size=n_samples),
        "PRG(ng/mL)": np.random.normal(0.5, 0.2, size=n_samples),
        "RBS(mg/dl)": np.random.normal(100, 20, size=n_samples),
        "Weight gain(Y/N)": np.random.choice([0, 1], size=n_samples),
        "hair growth(Y/N)": np.random.choice([0, 1], size=n_samples),
        "Skin darkening (Y/N)": np.random.choice([0, 1], size=n_samples),
        "Hair loss(Y/N)": np.random.choice([0, 1], size=n_samples),
        "Pimples(Y/N)": np.random.choice([0, 1], size=n_samples),
        "Fast food (Y/N)": np.random.choice([0, 1], size=n_samples),
        "Reg.Exercise(Y/N)": np.random.choice([0, 1], size=n_samples),
        "BP _Systolic (mmHg)": np.random.randint(100, 140, size=n_samples),
        "BP _Diastolic (mmHg)": np.random.randint(60, 90, size=n_samples),
        "Follicle No. (L)": np.random.randint(1, 20, size=n_samples),
        "Follicle No. (R)": np.random.randint(1, 20, size=n_samples),
        "Avg. F size (L) (mm)": np.random.normal(14, 4, size=n_samples),
        "Avg. F size (R) (mm)": np.random.normal(14, 4, size=n_samples),
        "Endometrium (mm)": np.random.normal(8, 2.5, size=n_samples)
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"[Notice] Generated sample synthetic dataset at: {file_path}")

def run_pipeline():
    print("=" * 70)
    print(" PCOS Prediction: Two-Stage Feature Selection + Ensemble + XAI ")
    print("=" * 70)

    # 1. Locate Dataset
    data_files = glob.glob("data/*.xlsx") + glob.glob("data/*.xls") + glob.glob("data/*.csv")
    if not data_files:
        demo_path = os.path.join("data", "PCOS_synthetic_demo.csv")
        create_synthetic_demo_data(demo_path)
        data_path = demo_path
    else:
        data_path = data_files[0]
        print(f"Found dataset: {data_path}")

    # 2. Data Preprocessing & Leakage-Free SMOTE
    X_train, X_test, y_train, y_test, all_features, scaler = load_and_preprocess_data(data_path)

    # 3. Two-Stage Feature Selection (Chi-Square -> CatBoost RFE -> 16 Features)
    selected_features, ranking = two_stage_feature_selection(
        X_train, y_train, stage1_k=30, stage2_k=16, wrapper_type="catboost"
    )

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # 4. Evaluate Baseline Classifiers
    print("\n" + "=" * 50)
    print(" Evaluating Baseline Models (16 Selected Features) ")
    print("=" * 50)
    base_models = get_base_models()
    results = []

    for name, model in base_models.items():
        res = evaluate_model(model, X_train_sel, y_train, X_test_sel, y_test)
        results.append({
            "Model": name,
            "Accuracy (%)": f"{res['Accuracy']:.2f}",
            "Precision (%)": f"{res['Precision']:.2f}",
            "Recall (%)": f"{res['Recall']:.2f}",
            "F1-Score (%)": f"{res['F1-Score']:.2f}"
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # 5. Evaluate Proposed Champion Ensemble (XGBoost + MLP Soft Voting)
    print("\n" + "=" * 50)
    print(" Evaluating Champion Hybrid Ensemble (XGBoost + MLP) ")
    print("=" * 50)
    champion = get_champion_ensemble()
    champ_eval = evaluate_model(champion, X_train_sel, y_train, X_test_sel, y_test)
    print(f"-> Accuracy:  {champ_eval['Accuracy']:.2f}%")
    print(f"-> Precision: {champ_eval['Precision']:.2f}%")
    print(f"-> Recall:    {champ_eval['Recall']:.2f}%")
    print(f"-> F1-Score:  {champ_eval['F1-Score']:.2f}%")
    print("\nConfusion Matrix:")
    print(champ_eval['Confusion Matrix'])

    # 6. 10-Fold Stratified Cross-Validation
    print("\nRunning 10-Fold Stratified Cross-Validation on Champion Model...")
    cv_scores = perform_10fold_cv(champion, X_train_sel, y_train)
    for metric, score in cv_scores.items():
        print(f" - {metric}: {score}")

    # 7. Explainable AI (SHAP & LIME)
    output_dir = "outputs"
    try:
        generate_shap_explanations(champion, X_train_sel, X_test_sel, output_dir=output_dir)
        
        # LIME for patient 0
        if len(X_test_sel) > 0:
            generate_lime_explanation(champion, X_train_sel, X_test_sel.iloc[0], instance_idx=0, output_dir=output_dir)
    except Exception as e:
        print(f"\n[Note on XAI]: SHAP/LIME generation skipped or encountered dependency note: {e}")

    print("\n" + "=" * 70)
    print(" Pipeline execution finished successfully! ")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
