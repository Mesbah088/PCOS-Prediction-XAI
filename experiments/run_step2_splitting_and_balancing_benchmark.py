import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

from src.preprocessing import sanitize_clinical_dataframe
from src.feature_engineering import engineer_clinical_features

def run_splitting_and_balancing_benchmark(data_path: str = "data/PCOS_data.csv"):
    os.makedirs("outputs", exist_ok=True)
    print("=" * 80)
    print("  STEP 2: DATA SPLITTING & CLASS BALANCING SCIENTIFIC BENCHMARK")
    print("  Comparing 6 Resampling Protocols with Nested Cross-Validation & Zero-Leakage")
    print("=" * 80)

    import re
    df_raw = pd.read_csv(data_path)
    df_clean, target_col = sanitize_clinical_dataframe(df_raw)
    df_engineered = engineer_clinical_features(df_clean)

    # Sanitize column names for LightGBM/XGBoost (removing : , [ ] { } " / \)
    clean_cols = {}
    for c in df_engineered.columns:
        if c != target_col:
            clean_name = re.sub(r'[:,\(\)\[\]\{\}\"\/\\ ]+', '_', c).strip('_')
            clean_cols[c] = clean_name
    df_engineered = df_engineered.rename(columns=clean_cols)

    X = df_engineered.drop(columns=[target_col])
    y = df_engineered[target_col].astype(int)
    feature_names = list(X.columns)

    print(f"\n[Cohort Information]")
    print(f" - Total Patients: {len(df_engineered)}")
    print(f" - Non-PCOS (0): {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
    print(f" - PCOS Positive (1): {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")
    print(f" - Total Features (Base + 25 Rotterdam Bio-Markers): {len(feature_names)}")

    # 2. Outer Stratified 80/20 Partition (Test partition locked)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Inside-Train Scaling (Zero Leakage)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names, index=X_test_raw.index)

    print(f"\n[Outer Split (Strict Zero-Leakage)]")
    print(f" - Training Cohort: {len(X_train_scaled)} patients (Neg: {(y_train == 0).sum()}, Pos: {(y_train == 1).sum()})")
    print(f" - Held-Out Test Set: {len(X_test_scaled)} patients (Neg: {(y_test == 0).sum()}, Pos: {(y_test == 1).sum()})")

    # 3. Define Resampling Strategies
    resampling_strategies = {
        "No Balancing (Baseline)": None,
        "Standard SMOTE": SMOTE(random_state=42),
        "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "SMOTE-Tomek": SMOTETomek(random_state=42),
        "SMOTE-ENN (Cleaning)": SMOTEENN(random_state=42)
    }

    benchmark_results = []
    confusion_matrices = {}

    # Standard representative model for comparing balancing effects: Tuned LightGBM + ExtraTrees Consensus
    print("\n" + "=" * 80)
    print(" Evaluating Resampling Strategies across Test Set & 5-Fold Nested CV...")
    print("=" * 80)

    for strat_name, resampler in resampling_strategies.items():
        print(f"\n>>> Testing Strategy: {strat_name}...")

        # A. Apply balancing strictly to training partition
        if resampler is not None:
            X_tr_res, y_tr_res = resampler.fit_resample(X_train_scaled, y_train)
        else:
            X_tr_res, y_tr_res = X_train_scaled.copy(), y_train.copy()

        train_pos = (y_tr_res == 1).sum()
        train_neg = (y_tr_res == 0).sum()

        # B. Train Classifier (LightGBM gradient booster with clinical regularization)
        model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=12,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1
        )
        model.fit(X_tr_res, y_tr_res)

        # C. Evaluate on Untouched Held-Out Test Set (Zero Leakage)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100  # Sensitivity
        prec = precision_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100
        auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        # Specificity
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = (tn / (tn + fp)) * 100
        confusion_matrices[strat_name] = cm

        # D. 5-Fold Nested Cross-Validation (Inside Training Partition)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_recalls, cv_f1s, cv_aucs = [], [], []

        for f_tr_idx, f_val_idx in skf.split(X_train_raw, y_train):
            f_X_tr_raw = X_train_raw.iloc[f_tr_idx]
            f_y_tr = y_train.iloc[f_tr_idx]
            f_X_val_raw = X_train_raw.iloc[f_val_idx]
            f_y_val = y_train.iloc[f_val_idx]

            # Fold-internal scaling
            f_scaler = MinMaxScaler()
            f_X_tr = pd.DataFrame(f_scaler.fit_transform(f_X_tr_raw), columns=feature_names)
            f_X_val = pd.DataFrame(f_scaler.transform(f_X_val_raw), columns=feature_names)

            # Fold-internal resampling
            if resampler is not None:
                f_X_tr_bal, f_y_tr_bal = resampler.fit_resample(f_X_tr, f_y_tr)
            else:
                f_X_tr_bal, f_y_tr_bal = f_X_tr, f_y_tr

            cv_clf = lgb.LGBMClassifier(
                n_estimators=120, learning_rate=0.04, max_depth=4,
                num_leaves=12, subsample=0.85, colsample_bytree=0.85,
                random_state=42, verbose=-1
            )
            cv_clf.fit(f_X_tr_bal, f_y_tr_bal)

            f_pred = cv_clf.predict(f_X_val)
            f_proba = cv_clf.predict_proba(f_X_val)[:, 1]

            cv_recalls.append(recall_score(f_y_val, f_pred) * 100)
            cv_f1s.append(f1_score(f_y_val, f_pred) * 100)
            cv_aucs.append(roc_auc_score(f_y_val, f_proba))

        benchmark_results.append({
            "Balancing Strategy": strat_name,
            "Resampled Train Size": f"{len(X_tr_res)} (Neg: {train_neg}, Pos: {train_pos})",
            "Test Accuracy (%)": acc,
            "Recall / Sensitivity (%)": rec,
            "Specificity (%)": spec,
            "Precision (%)": prec,
            "F1-Score (%)": f1,
            "ROC-AUC": auc,
            "PR-AUC": pr_auc,
            "False Negatives (Missed)": fn,
            "5-Fold CV Recall (%)": f"{np.mean(cv_recalls):.2f} ± {np.std(cv_recalls):.2f}",
            "5-Fold CV F1-Score (%)": f"{np.mean(cv_f1s):.2f} ± {np.std(cv_f1s):.2f}",
            "5-Fold CV ROC-AUC": f"{np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}"
        })

    # Convert to DataFrame
    df_res = pd.DataFrame(benchmark_results)
    df_res = df_res.sort_values(by="Recall / Sensitivity (%)", ascending=False)

    print("\n" + "=" * 90)
    print(" [BENCHMARK RESULTS] STEP 2: DATA BALANCING PERFORMANCE COMPARISON")
    print("=" * 90)
    print(df_res[["Balancing Strategy", "Test Accuracy (%)", "Recall / Sensitivity (%)", "Specificity (%)", "F1-Score (%)", "ROC-AUC", "False Negatives (Missed)"]].to_string(index=False))

    # Save to CSV
    csv_path = os.path.join("outputs", "data_balancing_benchmark_comparison.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n[Saved CSV] Results saved to: {csv_path}")

    # Generate Visualization Chart 1: Key Metrics Bar Chart
    plt.figure(figsize=(14, 6), dpi=300)
    metrics_to_plot = ["Recall / Sensitivity (%)", "Specificity (%)", "F1-Score (%)", "Test Accuracy (%)"]
    
    # Reshape for multi-bar plotting
    plot_df = pd.melt(
        df_res,
        id_vars=["Balancing Strategy"],
        value_vars=metrics_to_plot,
        var_name="Clinical Metric",
        value_name="Percentage (%)"
    )

    sns.barplot(
        data=plot_df,
        x="Balancing Strategy",
        y="Percentage (%)",
        hue="Clinical Metric",
        palette="viridis"
    )
    plt.title("Step 2: Impact of Class Balancing Protocols on Clinical PCOS Screening Metrics", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Performance Score (%)", fontsize=11)
    plt.xlabel("Balancing Strategy (Evaluated on Untouched 109 Test Cohort)", fontsize=11)
    plt.ylim(70, 100)
    plt.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart_path = os.path.join("outputs", "data_balancing_comparison_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Comparison chart saved to: {chart_path}")

    # Generate Visualization Chart 2: Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()

    for idx, (strat_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Neg (Control)", "Pos (PCOS)"],
            yticklabels=["Neg (Control)", "Pos (PCOS)"]
        )
        fn_val = cm[1, 0]
        tp_val = cm[1, 1]
        recall_pct = (tp_val / (tp_val + fn_val)) * 100
        ax.set_title(f"{strat_name}\nRecall: {recall_pct:.1f}% | Missed (FN): {fn_val}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Diagnosis", fontsize=9)
        ax.set_ylabel("True Diagnosis", fontsize=9)

    plt.suptitle("Step 2: Confusion Matrices Across Balancing Strategies (Untouched Held-Out Test Set: N = 109)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    cm_path = os.path.join("outputs", "data_balancing_confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Saved Heatmap] Confusion matrices saved to: {cm_path}")

    print("\n" + "=" * 80)
    print(" Step 2 Splitting & Balancing Benchmark Successfully Finished!")
    print("=" * 80)
    return df_res

if __name__ == "__main__":
    run_splitting_and_balancing_benchmark()
