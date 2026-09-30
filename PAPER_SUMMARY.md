# 📄 Research Paper Summary

## Title
**A Two-Stage Hybrid Feature Selection and Ensemble Learning Framework With Explainable AI for Accurate PCOS Prediction**

- **Journal:** *Health Science Reports*, Wiley (2026); Vol. 9: e73100
- **DOI:** [10.1002/hsr2.73100](https://doi.org/10.1002/hsr2.73100)
- **Authors:** Md. Rakibul Hasan Efty, Md. Naymur Rohman, Khandaker Mohammad Mohi Uddin, Md. Mahbubur Rahman, Md. Ashraf Uddin
- **Affiliations:** 
  1. Department of Computer Science and Engineering, Southeast University, Dhaka, Bangladesh
  2. Department of Electrical Engineering and Computer Science, University of Wyoming, USA
  3. School of Information and Technology, Deakin University, Australia
- **Data/Code Repository:** [GitHub - KMM-Uddin/XAI_PCOS-Prediction](https://github.com/KMM-Uddin/XAI_PCOS-Prediction.git)

---

## 1. 🎯 Background & Motivation

- **Clinical Problem:** Polycystic Ovary Syndrome (PCOS) is a common endocrine disorder affecting 1 in 10 women of reproductive age, often leading to infertility, metabolic complications, and diabetes. Around 70% of cases worldwide remain undiagnosed due to overlapping symptoms and resource-intensive diagnostic procedures.
- **Goal:** Develop an accurate, data-efficient, non-overfitting, and clinically interpretable Machine Learning Decision Support System (CDSS) for early PCOS screening.

---

## 2. 📊 Dataset Summary

- **Source:** Kaggle PCOS Dataset (collected from 10 distinct hospitals in Kerala, India).
- **Size:** 541 female patient records.
- **Attributes:** 44 initial clinical and metabolic features.
- **Class Balance:**
  - **Non-PCOS (0):** 364 patients (67.3%)
  - **PCOS (1):** 177 patients (32.7%)

---

## 3. ⚙️ Preprocessing & Leakage Prevention Pipeline

1. **Cleaning & Imputation:**
   - Redundant identifiers removed: `Sl. No.`, `Patient File No.`, and empty columns (`Unnamed: 44`).
   - Missing value imputation: Mode imputation applied for sparse columns like `Marriage Status (Yrs)` and `Fast Food (Y/N)`.
2. **Feature Normalization:**
   - MinMax Scaling rescales all features into $[0, 1]$ interval.
3. **Strict Data Leakage Prevention:**
   - 80/20 train-test split applied **before** oversampling and feature selection.
   - **SMOTE** (Synthetic Minority Oversampling Technique) was applied **exclusively on the 80% training set** (expanded from 432 to 582 balanced instances). The 20% test set (109 samples) remained completely untouched.

---

## 4. 🔍 Two-Stage Hybrid Feature Selection

The paper addresses high dimensionality and multicollinearity using a two-stage filter-wrapper consensus:

```mermaid
flowchart LR
    A["Original Dataset (40 Features)"] --> B["Stage 1: Chi-Square Filter (SelectKBest)"]
    B --> C["Reduced Set (30 Features)"]
    C --> D["Stage 2: RFE Wrapper (CatBoost)"]
    D --> E["Final Critical Set (16 Features)"]
```

### 📋 The 16 Selected Critical Features (CatBoost RFE)
1. `Follicle (R)` – Number of follicles in right ovary *(strongest predictor)*
2. `Follicle (L)` – Number of follicles in left ovary
3. `Hair growth (Y/N)` – Hirsutism indicator
4. `Cycle (R/I)` – Menstrual regularity (Regular/Irregular)
5. `Weight gain (Y/N)`
6. `Skin darkening (Y/N)` – Acanthosis nigricans
7. `Cycle length (days)`
8. `Pimples (Y/N)`
9. `Avg. F size (L) (mm)` – Average follicle diameter (Left)
10. `Marriage status (years)`
11. `Avg. F size (R) (mm)` – Average follicle diameter (Right)
12. `RR (breaths/min)` – Respiration rate
13. `Fast food (Y/N)` – Dietary habits
14. `Weight (kg)`
15. `FSH (mIU/mL)` – Follicle-Stimulating Hormone
16. `AMH (ng/mL)` – Anti-Müllerian Hormone

---

## 5. 🤖 Machine Learning Models & Hybrid Ensemble

### Evaluated Base Classifiers (11 Models)
Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), XGBoost (XGB), CatBoost (CB), AdaBoost (AB), Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), Linear Discriminant Analysis (LDA), and Gaussian Naive Bayes (GNB).

### 🏆 Champion Ensemble Architecture
- **Soft-Voting Ensemble:** Combining a tree-based gradient booster (**XGBoost**) with a deep representation neural network (**Multi-Layer Perceptron / MLP**).
- **Rationale:** XGBoost captures complex non-linear feature splits, while MLP captures smooth continuous hyperplanes; soft-voting averages class probability predictions.

---

## 6. 📈 Key Results & Metrics

| Model Setup | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---|:---:|:---:|:---:|:---:|
| **XGBoost + MLP (Champion Ensemble)** | **96.33%** | **96.35%** | **96.33%** | **96.30%** |
| XGBoost + CatBoost (Ensemble) | 95.41% | 95.47% | 95.41% | 95.36% |
| Standalone XGBoost | 94.50% | 94.49% | 94.50% | 94.45% |
| Standalone MLP | 94.50% | 94.50% | 94.50% | 94.50% |
| Logistic Regression | 93.58% | 93.64% | 93.58% | 93.60% |
| CatBoost | 92.66% | 92.62% | 92.66% | 92.61% |
| Random Forest | 90.83% | 90.83% | 90.83% | 90.83% |
| SVM | 90.83% | 91.31% | 90.83% | 90.94% |

### Generalization & Overfitting Validation
- **10-Fold Stratified Cross-Validation:** XGB + MLP achieved **92.04% ± 2.58%** accuracy and **92.03% ± 2.59%** F1-score across folds.
- **Overfitting Gap:** The two-stage feature selection reduced the train-test overfitting margin from **7.57%** down to **2.98%**.

---

## 7. 💡 Explainable AI (XAI) Framework

### 1. SHAP (SHapley Additive exPlanations) - *Global Interpretability*
- Quantifies global feature importance and positive/negative contribution directions.
- **Top Positive Drivers of PCOS:** High Follicle counts (R & L), Irregular Cycle (`Cycle (R/I)`), Hair Growth, and Weight Gain.
- **Low Risk Markers:** Low follicle counts, normal cycles, and low AMH.

### 2. LIME (Local Interpretable Model-Agnostic Explanations) - *Local Interpretability*
- Explains single-patient predictions for clinicians.
- Assigns percentage weight contributions to each biomarker to build medical trust.

---

## 8. ⚡ Computational Complexity

| Stage | Operation | Theoretical Complexity |
|---|---|---|
| **Stage 1 (Chi-Square)** | Univariate dependence testing | $\mathcal{O}(m \times n)$ |
| **Stage 2 (CatBoost RFE)** | Iterative elimination & retraining | $\mathcal{O}((p - k) \times n \times t \times \log n)$ |
| **Final Ensemble Training** | Training & Evaluation | $\mathcal{O}(n \times t \times \log n)$ |

*Where $n$ = samples, $m$ = original features (40), $p$ = intermediate features (30), $k$ = final features (16), $t$ = number of trees.*

---

## 9. ⚠️ Limitations & Future Direction

1. **Demographic Bias:** Trained on 541 patients from Kerala, India. Multi-center, multi-ethnic clinical validation is required.
2. **Clinical Role:** Designed as a Decision Support System (CDSS) to triage and flag high-risk cases for pelvic ultrasound/hormonal panels, not as a replacement for clinicians.
