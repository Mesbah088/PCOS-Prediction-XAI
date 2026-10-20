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
import joblib

from src.preprocessing import load_and_preprocess_data
from src.xai import (
    compute_360_shap_explanations,
    generate_lime_patient_explanation,
    generate_actionable_dice_counterfactual,
    generate_clinical_scoring_nomogram
)

def sanitize_name(col: str) -> str:
    return re.sub(r'[:,\(\)\[\]\{\}\"\/\\ ]+', '_', col).strip('_')

def run_step5_pipeline():
    os.makedirs("outputs", exist_ok=True)
    print("=" * 85, flush=True)
    print("  STEP 5: 360° CLINICAL EXPLAINABLE AI (SHAP, LIME, DiCE) & CLINICAL NOMOGRAM", flush=True)
    print("  Interpreting Calibrated Champion Super-Ensemble on Locked Test Patients", flush=True)
    print("=" * 85, flush=True)

    # 1. Load Serialized Champion Model
    model_path = os.path.join("outputs", "champion_ensemble_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Serialized model not found at {model_path}. Run step 4 first.")
    
    saved_bundle = joblib.load(model_path)
    model = saved_bundle["model"]
    feature_names = saved_bundle["feature_names"]
    print(f"\n[Model Loaded] Serialized Calibrated Champion Model loaded successfully.", flush=True)
    print(f"Features ({len(feature_names)}): {feature_names}", flush=True)

    # 2. Ingest Data
    data_path = os.path.join("data", "PCOS_data.csv")
    X_train_raw, X_test_raw, y_train, y_test, _, _ = load_and_preprocess_data(
        file_path=data_path,
        apply_feature_engineering=True,
        resampling_strategy="smote_enn",
        test_size=0.20,
        random_state=42
    )

    col_map = {c: sanitize_name(c) for c in X_train_raw.columns}
    X_tr_all = X_train_raw.rename(columns=col_map)
    X_te_all = X_test_raw.rename(columns=col_map)

    X_train = X_tr_all[feature_names]
    X_test = X_te_all[feature_names]

    # 3. Tier 1 & Tier 2: Compute 360° SHAP Explanations
    shap_results = compute_360_shap_explanations(
        champion_dict=saved_bundle,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir="outputs"
    )

    # 4. Tier 2: LIME Local Surrogate Explanation
    high_risk_idx = shap_results["high_risk_idx"]
    high_risk_patient = X_test.iloc[high_risk_idx]
    generate_lime_patient_explanation(
        model=model,
        X_train=X_train,
        patient_instance=high_risk_patient,
        feature_names=feature_names,
        instance_idx=high_risk_idx,
        output_dir="outputs"
    )

    # 5. Tier 3: DiCE Actionable Clinical Counterfactuals
    cf_results = generate_actionable_dice_counterfactual(
        model=model,
        patient_row=high_risk_patient,
        feature_names=feature_names,
        target_risk_threshold=0.25,
        output_dir="outputs"
    )

    print("\n" + "=" * 75, flush=True)
    print(" [DiCE ACTIONABLE COUNTERFACTUAL PRESCRIPTION GENERATED]", flush=True)
    print("=" * 75, flush=True)
    print(f" Initial Patient Risk: {cf_results['original_risk']:.1f}% (Predicted Positive PCOS)", flush=True)
    print(f" Counterfactual Goal:  {cf_results['counterfactual_risk']:.1f}% (Healthy Negative Control)", flush=True)
    print(f" Absolute Risk Drop:   -{cf_results['risk_reduction']:.1f}%", flush=True)
    print("\n Clinically Realistic Interventions Prescribed by DiCE Engine:", flush=True)
    for idx, p in enumerate(cf_results["prescriptions"], 1):
        print(f"   {idx}. {p}", flush=True)

    # 6. Tier 4: Clinical Bedside Scoring Nomogram
    generate_clinical_scoring_nomogram(feature_names=feature_names, output_dir="outputs")

    print("\n" + "=" * 85, flush=True)
    print(" Step 5 XAI, Counterfactuals & Nomogram Pipeline Successfully Finished!", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    run_step5_pipeline()
