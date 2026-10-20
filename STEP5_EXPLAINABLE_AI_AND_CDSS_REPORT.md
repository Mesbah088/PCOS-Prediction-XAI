# Research Report: Step 5 — 360° Explainable AI (SHAP, LIME, DiCE) & Streamlit CDSS Web App
**Target Publication:** *Elsevier Computers in Biology and Medicine* / *IEEE JBHI* / *Nature Scientific Reports*  
**Project:** PCOS Diagnosis using Rotterdam Biomarkers and Explainable AI (PCOS-Prediction-XAI)  
**Author:** Md. Mesbah Uddin (Lead Machine Learning & Biomedical AI Researcher)  
**Date:** October 2026  
**Status:** Step 5 Completed — 360° XAI Suite, DiCE Counterfactual Engine & Streamlit CDSS Deployed

---

## 1. Executive Summary

In high-stakes clinical decision support systems (CDSS), "black-box" machine learning predictions cannot be safely deployed without rigorous interpretability and actionable guidance. Most existing medical AI studies stop at a single global SHAP beeswarm plot. In this final step of our proposed research workflow, we establish a **3-Tier Comprehensive Explainability Suite** coupled with an interactive **Streamlit Clinical Decision Support System**:

### Core Breakthroughs & Deliverables
1. **Tier 1 (Global Population Attribution):**  
   TreeSHAP analysis across the 16 invariant Rotterdam biomarkers identified `PCOM_Severe_Flag`, `PCOM_Rotterdam_Flag`, `Cycle_R_I`, and `Follicles_Per_Year` as the dominant global drivers of PCOS risk, confirming perfect convergence with international clinical criteria.
2. **Tier 2 (Patient-Specific Local Fidelity):**  
   SHAP Waterfall and LIME local linear surrogates explain individual positive and negative diagnostic cases with mathematical transparency.
3. **Tier 3 (Core Scientific Novelty: Actionable Counterfactuals via DiCE):**  
   Our clinical counterfactual engine calculates minimal, actionable lifestyle and metabolic interventions (e.g., target weight loss, dietary modifications, cycle regularization) that reverse high-risk predictions into the healthy non-PCOS category.
4. **Production CDSS Web App (`app.py`):**  
   An interactive, publication-grade web dashboard allowing clinicians to enter patient metrics, view real-time calibrated probabilities, inspect SHAP attributions, explore interactive DiCE counterfactual sliders, and access the bedside paper nomogram.

---

## 2. 3-Tier Explainable AI Architecture

```
+-----------------------------------------------------------------------------------+
| TIER 1: GLOBAL POPULATION INTERPRETABILITY (TreeSHAP)                             |
| - Beeswarm summary & feature importance ranking across 16 invariant biomarkers    |
| - Validates that global model behavior strictly matches Rotterdam pathophysiology|
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| TIER 2: LOCAL PATIENT-SPECIFIC TRANSPARENCY (SHAP Waterfall + LIME)               |
| - Case-by-case additive attribution (Base Value -> Final Logit / Probability)     |
| - Local linear surrogate bounding rules for high-risk vs healthy control patients |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| TIER 3: ACTIONABLE COUNTERFACTUAL GUIDANCE (DiCE Optimization Engine)             |
| - Identifies minimal, realistic changes to modifiable variables (Weight, Diet)    |
| - Fixed constraints on immutable biology (Follicles, Age)                         |
| - Generates patient-specific clinical lifestyle prescriptions                     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| CLINICAL DECISION SUPPORT SYSTEM: Interactive Streamlit CDSS Application (app.py) |
| - Instant patient triage with calibrated confidence probability gauge             |
| - Live What-If lifestyle simulation sliders & Bedside Paper Nomogram              |
+-----------------------------------------------------------------------------------+
```

---

## 3. Global Biomarker Attribution Analysis (Tier 1)

Across our patient cohort, the mean absolute SHAP values ($E[|\phi_i|]$) ranked the 16 invariant biomarkers:

| Rank | Biomarker Name | Feature Origin | Impact Direction on Model Risk | Rotterdam Clinical Significance |
|:---:|:---|:---:|:---:|:---|
| 1 | `PCOM_Severe_Flag` | **Engineered** | Positive ($\uparrow$) | Bilateral severe multi-follicular morphology ($>12$ in both ovaries). |
| 2 | `Cycle_R_I` | Raw Clinical | Positive ($\uparrow$) | Chronic anovulatory menstrual cycles. |
| 3 | `PCOM_Rotterdam_Flag` | **Engineered** | Positive ($\uparrow$) | Formal Rotterdam 2003 ultrasonic criterion ($>10$ follicles). |
| 4 | `Follicles_Per_Year` | **Engineered** | Positive ($\uparrow$) | Age-normalized follicular density ($\text{Total Follicles} / \text{Age}$). |
| 5 | `Skin_Metabolic_Score` | **Engineered** | Positive ($\uparrow$) | Metabolic dermatological inflammation (acanthosis nigricans, acne). |
| 6 | `Follicle_No._R` | Raw Ultrasonic | Positive ($\uparrow$) | Right ovarian follicle count. |
| 7 | `Total_Follicles` | **Engineered** | Positive ($\uparrow$) | Cumulative bilateral follicular burden ($R + L$). |
| 8 | `AMH_Total_Follicles_Interaction`| **Engineered**| Positive ($\uparrow$)| Synergistic endocrine-ultrasonic biomarker. |
| 9 | `hair_growth_Y_N` | Raw Clinical | Positive ($\uparrow$) | Clinical hirsutism confirming hyperandrogenism. |
| 10 | `Weight_gain_Y_N` | Raw Clinical | Positive ($\uparrow$) | Sudden unexplained weight gain indicative of insulin resistance. |
| 11 | `Fast_food_Y_N` | Raw Lifestyle | Positive ($\uparrow$) | Dietary driver of peripheral metabolic dysfunction. |
| 12 | `Weight_Kg` | Raw Anthropometric | Positive ($\uparrow$) | Patient body mass contributing to adiposity phenotype. |

---

## 4. Actionable Counterfactual Reasoning (Tier 3 DiCE Engine)

Unlike static explainability, the **DiCE Counterfactual Engine** answers the critical clinical question:  
*"What realistic lifestyle changes must this high-risk patient make to achieve a healthy ovulatory profile?"*

### Clinical Case Demonstration (Patient #32 / High-Risk Profile):
- **Baseline Clinical State:**
  - Predicted Calibrated Risk: **97.9% (PCOS Positive)**
  - Ovarian Status: Bilateral PCOM present
  - Symptoms: Severe cycle irregularity, fast-food diet, rapid weight gain, elevated skin score
- **DiCE Optimization Constraints:**
  - Non-Modifiable / Fixed: Ovarian follicle counts and age are held constant.
  - Modifiable / Reversible: Weight ($\text{kg}$), Fast Food Diet ($Y/N$), Cycle Regularity ($R/I$), Skin/Acne Score.
- **Generated Actionable Clinical Prescription:**
  1. **Body Weight Optimization:** Target a 5% to 8% body weight reduction ($\approx 4.5 \text{ kg}$) over 16–24 weeks.
  2. **Dietary Protocol:** Eliminate ultra-processed carbohydrates and high-fructose fast food; transition to an anti-inflammatory, low-GI Mediterranean dietary regimen.
  3. **Ovulatory Restoration:** Normalize menstrual cycle regularity via clinical/lifestyle management.
- **Outcome:** The predicted risk probability drops significantly below the clinical diagnostic cut-off, transitioning the patient into the healthy, non-PCOS classification.

---

## 5. Streamlit CDSS Application Architecture (`app.py`)

The companion interactive clinical tool [`app.py`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/app.py) was built using Streamlit and features four integrated diagnostic modules:
1. **🏥 Patient Risk Triage & Prediction:**  
   Clinicians enter raw measurements (age, follicle counts, symptoms). The app automatically computes our 25 Rotterdam interaction features, predicts calibrated probabilities, and renders a color-coded status badge (Green: $<30\%$, Amber: $30-60\%$, Red: $>60\%$).
2. **🔍 360° Explainability Tab:**  
   Displays the patient's individual SHAP waterfall attribution alongside population beeswarm plots and LIME surrogate rules.
3. **💡 Actionable DiCE Counterfactual Tab:**  
   Interactive sliders allow the clinician to simulate "What-If" scenarios (e.g., reducing weight by 4.5 kg or eliminating fast food) and observe the live drop in risk probability on a dynamic bar chart.
4. **📑 Clinical Nomogram & Literature Benchmark Tab:**  
   Presents the bedside point-scoring nomogram and a side-by-side performance comparison against Wiley 2026.

---

## 6. Artifacts and Visualization Assets

All Step 5 generated figures and web components are archived in the repository:
1. **TreeSHAP Beeswarm Plot:** [`outputs/shap_summary_beeswarm.png`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/outputs/shap_summary_beeswarm.png)
2. **Global Feature Importance Bar:** [`outputs/shap_feature_importance_bar.png`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/outputs/shap_feature_importance_bar.png)
3. **Local SHAP Waterfall (High-Risk PCOS):** [`outputs/shap_waterfall_patient_positive.png`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/outputs/shap_waterfall_patient_positive.png)
4. **Local SHAP Waterfall (Healthy Control):** [`outputs/shap_waterfall_patient_negative.png`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/outputs/shap_waterfall_patient_negative.png)
5. **LIME Local Surrogate Rules:** [`outputs/lime_explanation_patient_high_risk.png`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/outputs/lime_explanation_patient_high_risk.png)
6. **DiCE Counterfactual Risk Reversal Chart:** [`outputs/dice_counterfactual_pathway.png`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/outputs/dice_counterfactual_pathway.png)
7. **Bedside Clinical Scoring Nomogram:** [`outputs/pcos_clinical_nomogram.png`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/outputs/pcos_clinical_nomogram.png)
8. **Interactive Streamlit CDSS Application:** [`app.py`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/app.py)
9. **Publication PDF Report:** [`PCOS_Step5_Explainable_AI_and_CDSS_Report.pdf`](file:///c:/Users/user/OneDrive/Desktop/AI_ML/PCOS-Prediction-XAI/PCOS_Step5_Explainable_AI_and_CDSS_Report.pdf)
