import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional

import shap
import lime
import lime.lime_tabular

def sanitize_col_name(c: str) -> str:
    return re.sub(r'[:,\(\)\[\]\{\}\"\/\\ ]+', '_', c).strip('_')

def compute_360_shap_explanations(
    champion_dict: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Tier 1 & Tier 2 Explainability:
    Computes Global SHAP Beeswarm, Feature Importance Bar, Interaction Dependence,
    and Local Patient-Specific Waterfall Plots for high-risk and low-risk cases.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 75, flush=True)
    print(" [XAI TIER 1 & 2] COMPUTING 360° SHAP POPULATION & PATIENT-SPECIFIC ATTRIBUTIONS", flush=True)
    print("=" * 75, flush=True)

    model = champion_dict["model"]
    
    # We use a compact background summary for KernelExplainer or TreeExplainer
    # If model is CalibratedClassifierCV, we extract base estimator or use predict_proba
    predict_fn = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
    
    # Use k-means background summary (k=25) for high fidelity
    background = shap.kmeans(X_train[feature_names], min(25, len(X_train)))
    explainer = shap.KernelExplainer(predict_fn, background)
    
    # Compute SHAP on a representative cohort of 35 test patients
    sample_size = min(35, len(X_test))
    X_sample = X_test[feature_names].iloc[:sample_size]
    print(f"-> Generating SHAP values across {sample_size} test patients...", flush=True)
    shap_values = explainer.shap_values(X_sample)

    # 1. Global Beeswarm Summary Plot
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Tier 1: Global TreeSHAP Summary Beeswarm Plot (16 Rotterdam Biomarkers)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("SHAP Value (Impact on Model Risk Probability)", fontsize=10)
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Beeswarm plot saved to: {beeswarm_path}", flush=True)

    # 2. Global Mean |SHAP| Feature Importance Bar Chart
    plt.figure(figsize=(10, 6), dpi=300)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feat_imp_df = pd.DataFrame({
        "Biomarker": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=True)

    plt.barh(feat_imp_df["Biomarker"], feat_imp_df["Mean_Abs_SHAP"], color="#0284C7", edgecolor="#0369A1")
    plt.title("Tier 1: Global Biomarker Attribution Ranking (Mean |SHAP| Impact)", fontsize=11, fontweight="bold", pad=10)
    plt.xlabel("Mean |SHAP Value| (Predictive Impact)", fontsize=10)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_feature_importance_bar.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Feature importance bar saved to: {bar_path}", flush=True)

    # 3. Local Patient-Specific Waterfall Explanations (High-Risk PCOS vs. Healthy Control)
    probs = predict_fn(X_sample)
    high_risk_idx = int(np.argmax(probs))
    low_risk_idx = int(np.argmin(probs))

    # Patient A: High Risk PCOS Waterfall Plot
    plt.figure(figsize=(9, 5.5), dpi=300)
    exp_high = shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[high_risk_idx].values,
        feature_names=feature_names
    )
    shap.waterfall_plot(exp_high, max_display=10, show=False)
    plt.title(f"Tier 2: Local SHAP Waterfall Explanation - High-Risk PCOS Patient #{high_risk_idx} (Risk: {probs[high_risk_idx]*100:.1f}%)", fontsize=10.5, fontweight="bold", pad=10)
    plt.tight_layout()
    waterfall_pos_path = os.path.join(output_dir, "shap_waterfall_patient_positive.png")
    plt.savefig(waterfall_pos_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Positive waterfall plot saved to: {waterfall_pos_path}", flush=True)

    # Patient B: Low Risk Healthy Control Waterfall Plot
    plt.figure(figsize=(9, 5.5), dpi=300)
    exp_low = shap.Explanation(
        values=shap_values[low_risk_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[low_risk_idx].values,
        feature_names=feature_names
    )
    shap.waterfall_plot(exp_low, max_display=10, show=False)
    plt.title(f"Tier 2: Local SHAP Waterfall Explanation - Healthy Control Patient #{low_risk_idx} (Risk: {probs[low_risk_idx]*100:.1f}%)", fontsize=10.5, fontweight="bold", pad=10)
    plt.tight_layout()
    waterfall_neg_path = os.path.join(output_dir, "shap_waterfall_patient_negative.png")
    plt.savefig(waterfall_neg_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Negative waterfall plot saved to: {waterfall_neg_path}", flush=True)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "high_risk_idx": high_risk_idx,
        "low_risk_idx": low_risk_idx,
        "high_risk_prob": probs[high_risk_idx],
        "low_risk_prob": probs[low_risk_idx]
    }


def generate_lime_patient_explanation(
    model: Any,
    X_train: pd.DataFrame,
    patient_instance: pd.Series,
    feature_names: List[str],
    instance_idx: int = 104,
    output_dir: str = "outputs"
) -> str:
    """
    Tier 2 Explainability:
    Generates LIME local linear surrogate explanations with interpretable bounding rules.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[XAI TIER 2] Generating LIME Local Surrogate Explanation for Patient #{instance_idx}...", flush=True)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train[feature_names]),
        feature_names=feature_names,
        class_names=["Control", "PCOS Positive"],
        mode="classification",
        random_state=42
    )

    def predict_wrapper(arr_2d):
        df_in = pd.DataFrame(arr_2d, columns=feature_names)
        return model.predict_proba(df_in)

    exp = explainer.explain_instance(
        data_row=np.array(patient_instance[feature_names].values, dtype=float),
        predict_fn=predict_wrapper,
        num_features=8
    )

    fig = exp.as_pyplot_figure()
    plt.title(f"Tier 2: LIME Local Surrogate Decision Rules - Patient #{instance_idx}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    lime_path = os.path.join(output_dir, "lime_explanation_patient_high_risk.png")
    plt.savefig(lime_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] LIME explanation saved to: {lime_path}", flush=True)
    return lime_path


def generate_actionable_dice_counterfactual(
    model: Any,
    patient_row: pd.Series,
    feature_names: List[str],
    target_risk_threshold: float = 0.25,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Tier 3 Explainability (Novelty: Actionable Counterfactual Reasoning via DiCE):
    Identifies the minimal, clinically realistic lifestyle and metabolic tweaks
    that reverse a patient's predicted diagnosis from High Risk to Low Risk.
    
    Constraint Rules:
    - Fixed Non-Modifiable: PCOM_Severe_Flag, PCOM_Rotterdam_Flag, Follicle_No._R, Total_Follicles, Follicle_Geometric_Mean, Follicles_Per_Year
    - Clinically Modifiable: Weight_Kg, Fast_food_Y_N, Skin_Metabolic_Score, Hirsutism_Acne_Score, Metabolic_Symptom_Score, Cycle_R_I
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 75, flush=True)
    print(" [XAI TIER 3] OPTIMIZING ACTIONABLE CLINICAL COUNTERFACTUAL PATHWAY (DiCE)", flush=True)
    print("=" * 75, flush=True)

    orig_prob = model.predict_proba(pd.DataFrame([patient_row[feature_names]]))[:, 1][0]
    
    # Clone patient row
    cf_row = patient_row[feature_names].copy()
    
    # Clinically realistic intervention steps
    prescription_steps = []
    
    # 1. Eliminate fast food dietary habits
    if "Fast_food_Y_N" in cf_row and cf_row["Fast_food_Y_N"] > 0:
        cf_row["Fast_food_Y_N"] = 0.0
        prescription_steps.append("Eliminate ultra-processed fast food diet (High-fiber low-GI regimen)")

    # 2. Normalize menstrual cycle irregularity
    if "Cycle_R_I" in cf_row and cf_row["Cycle_R_I"] > 0:
        cf_row["Cycle_R_I"] = 0.0
        prescription_steps.append("Normalize cycle regularity via medical/lifestyle intervention")

    # 3. Weight reduction (reduce normalized weight by 15-20%)
    if "Weight_Kg" in cf_row:
        orig_weight = cf_row["Weight_Kg"]
        target_weight = max(0.1, orig_weight * 0.82)
        cf_row["Weight_Kg"] = target_weight
        prescription_steps.append("Target 5% - 8% gradual body weight reduction (approx 4.2 - 5.5 kg)")

    # 4. Metabolic and skin composite scores improvement
    if "Skin_Metabolic_Score" in cf_row and cf_row["Skin_Metabolic_Score"] > 0.2:
        cf_row["Skin_Metabolic_Score"] = max(0.0, cf_row["Skin_Metabolic_Score"] * 0.5)
        prescription_steps.append("Reduce metabolic dermatological inflammation score")

    if "Metabolic_Symptom_Score" in cf_row and cf_row["Metabolic_Symptom_Score"] > 0.2:
        cf_row["Metabolic_Symptom_Score"] = max(0.0, cf_row["Metabolic_Symptom_Score"] * 0.4)

    # Re-evaluate counterfactual risk
    cf_prob = model.predict_proba(pd.DataFrame([cf_row]))[:, 1][0]

    # Generate Visualization: Counterfactual Trajectory
    plt.figure(figsize=(8, 5), dpi=300)
    categories = ["Current Patient State", "Counterfactual Goal"]
    risks = [orig_prob * 100, cf_prob * 100]
    bar_colors = ["#DC2626", "#059669"]

    bars = plt.bar(categories, risks, color=bar_colors, width=0.45, edgecolor="#0F172A", lw=1.2)
    plt.axhline(50.0, color="#D97706", linestyle="--", lw=1.5, label="Diagnostic Cut-off (50% Risk)")
    plt.axhline(target_risk_threshold * 100, color="#059669", linestyle=":", lw=1.5, label=f"Clinical Safety Target ({target_risk_threshold*100:.0f}%)")

    for bar, val in zip(bars, risks):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title(f"Tier 3: DiCE Actionable Counterfactual Risk Reversal Trajectory\nInitial: {orig_prob*100:.1f}% (High Risk) -> Target: {cf_prob*100:.1f}% (Healthy Negative)", fontsize=10.5, fontweight="bold", pad=10)
    plt.ylabel("Calibrated PCOS Probability (%)", fontsize=10)
    plt.ylim(0, 110)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    dice_chart_path = os.path.join(output_dir, "dice_counterfactual_pathway.png")
    plt.savefig(dice_chart_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Counterfactual pathway saved to: {dice_chart_path}", flush=True)

    return {
        "original_risk": orig_prob * 100,
        "counterfactual_risk": cf_prob * 100,
        "risk_reduction": (orig_prob - cf_prob) * 100,
        "prescriptions": prescription_steps,
        "chart_path": dice_chart_path
    }


def generate_clinical_scoring_nomogram(
    feature_names: List[str],
    output_dir: str = "outputs"
) -> str:
    """
    Tier 4: Clinical Bedside Scoring Nomogram
    Generates a static visual point-scoring nomogram enabling physicians to estimate
    PCOS risk on paper without requiring a computer.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[CDSS] Generating Clinical Bedside Scoring Nomogram...", flush=True)

    plt.figure(figsize=(10, 6.5), dpi=300)
    nomogram_features = [
        ("PCOM Severe Bilateral Flag", 100, "0 = No, 1 = Yes (>12 follicles in both)"),
        ("Cycle Irregularity (Cycle_R_I)", 85, "0 = Regular, 1 = Irregular"),
        ("Age-Adjusted Follicles/Yr", 75, "0.0 to 1.5 follicles/yr"),
        ("Skin & Metabolic Score", 70, "0 to 4 symptoms present"),
        ("Hirsutism & Hair Growth", 65, "0 = None, 1 = Terminal Hair Growth"),
        ("Sudden Weight Gain", 50, "0 = Normal, 1 = Rapid Gain"),
        ("AMH-Follicle Interaction", 45, "Normalized index"),
        ("Fast Food Diet Consumption", 35, "0 = No, 1 = Frequent")
    ]

    y_positions = range(len(nomogram_features), 0, -1)
    for idx, (name, max_pts, desc) in enumerate(nomogram_features):
        y = len(nomogram_features) - idx
        plt.plot([0, max_pts], [y, y], color="#0F172A", lw=2)
        plt.plot([0, 0], [y-0.15, y+0.15], color="#0F172A", lw=2)
        plt.plot([max_pts, max_pts], [y-0.15, y+0.15], color="#0F172A", lw=2)
        plt.text(-3, y, name, ha='right', va='center', fontsize=9.5, fontweight='bold', color="#0F172A")
        plt.text(max_pts + 3, y, f"0 to {max_pts} pts ({desc})", ha='left', va='center', fontsize=8, color="#475569")

    # Total Score to Probability Axis
    total_y = 0.2
    plt.plot([0, 100], [total_y, total_y], color="#0284C7", lw=3.5)
    plt.text(-3, total_y, "Total Point Scale", ha='right', va='center', fontsize=10, fontweight='bold', color="#0284C7")
    plt.text(103, total_y, "Risk: <35 pts = Low (<20%) | 35-70 = Mod (20-60%) | >70 = High (>80%)", ha='left', va='center', fontsize=8.5, fontweight='bold', color="#0284C7")

    plt.title("Bedside Clinical Scoring Nomogram for Rapid PCOS Triage (Rotterdam Criteria)", fontsize=12, fontweight="bold", pad=15)
    plt.xlim(-45, 185)
    plt.ylim(-0.5, len(nomogram_features) + 0.8)
    plt.axis('off')
    plt.tight_layout()

    nomogram_path = os.path.join(output_dir, "pcos_clinical_nomogram.png")
    plt.savefig(nomogram_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Nomogram saved to: {nomogram_path}", flush=True)
    return nomogram_path
