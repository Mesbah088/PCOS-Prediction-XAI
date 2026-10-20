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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from src.preprocessing import load_and_preprocess_data
from src.feature_selection import tri_stage_hybrid_feature_selection, PAPER_16_FEATURES

def sanitize_cols(columns):
    return [re.sub(r'[:,\(\)\[\]\{\}\"\/\\ ]+', '_', c).strip('_') for c in columns]

def run_step3_benchmark():
    os.makedirs("outputs", exist_ok=True)
    print("=" * 80)
    print("  STEP 3: TRI-STAGE HYBRID FEATURE SELECTION (Tri-HFS) & ABLATION STUDY")
    print("  66 Features -> Stage 1 Filter (30) -> Stage 2 Wrapper (18) -> Stage 3 Stability (16)")
    print("=" * 80)

    # 1. Zero-Leakage Data Ingestion (SMOTE-ENN balancing on train fold)
    data_path = os.path.join("data", "PCOS_data.csv")
    X_train, X_test, y_train, y_test, all_features, scaler = load_and_preprocess_data(
        file_path=data_path,
        apply_feature_engineering=True,
        resampling_strategy="smote_enn",
        test_size=0.20,
        random_state=42
    )

    # Sanitize feature names for LightGBM
    col_map = {c: re.sub(r'[:,\(\)\[\]\{\}\"\/\\ ]+', '_', c).strip('_') for c in X_train.columns}
    X_train = X_train.rename(columns=col_map)
    X_test = X_test.rename(columns=col_map)

    # 2. Run Tri-Stage Hybrid Feature Selection
    final_16_feats, fs_meta = tri_stage_hybrid_feature_selection(
        X_train, y_train, stage1_k=30, stage2_k=18, stage3_k=16, wrapper_type="catboost", random_state=42
    )

    stage1_30_feats = fs_meta["stage1_features"]
    stage2_18_feats = fs_meta["stage2_features"]
    stage3_16_feats = final_16_feats

    # Map paper 16 features to sanitized names
    paper_16_sanitized = [col_map[c] for c in PAPER_16_FEATURES if c in col_map]

    # 3. Scientific Ablation Study Across Reduction Stages
    feature_sets = {
        "Level 0: All Features (No Selection)": list(X_train.columns),
        "Level 1: Stage 1 Filter (Chi2 + ANOVA)": stage1_30_feats,
        "Level 2: Stage 2 Wrapper (CatBoost RFE)": stage2_18_feats,
        "Level 3: Tri-HFS Consensus (Proposed Final)": stage3_16_feats,
        "Paper Baseline: Raw 16 Features (Wiley 2026)": paper_16_sanitized
    }

    ablation_results = []
    print("\n" + "=" * 80)
    print(" RUNNING SCIENTIFIC ABLATION BENCHMARK ACROSS FEATURE REDUCTION STAGES")
    print("=" * 80)

    for set_name, feats in feature_sets.items():
        # Subset features
        avail_feats = [f for f in feats if f in X_train.columns]
        X_tr_sub = X_train[avail_feats]
        X_te_sub = X_test[avail_feats]

        # Clinical ensemble model (Tuned LightGBM + CatBoost soft vote)
        m1 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=12, random_state=42, verbose=-1)
        m2 = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=4, random_seed=42, verbose=0)
        
        m1.fit(X_tr_sub, y_train)
        m2.fit(X_tr_sub, y_train)

        p1 = m1.predict_proba(X_te_sub)[:, 1]
        p2 = m2.predict_proba(X_te_sub)[:, 1]
        y_proba = 0.5 * p1 + 0.5 * p2
        y_pred = (y_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100
        auc = roc_auc_score(y_test, y_proba)

        # 5-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1s = []
        for f_tr, f_val in skf.split(X_tr_sub, y_train):
            cv_m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=12, random_state=42, verbose=-1)
            cv_m.fit(X_tr_sub.iloc[f_tr], y_train.iloc[f_tr])
            val_p = cv_m.predict(X_tr_sub.iloc[f_val])
            cv_f1s.append(f1_score(y_train.iloc[f_val], val_p) * 100)

        ablation_results.append({
            "Feature Selection Level": set_name,
            "Feature Count": len(avail_feats),
            "Test Accuracy (%)": acc,
            "Recall / Sensitivity (%)": rec,
            "Precision (%)": prec,
            "F1-Score (%)": f1,
            "ROC-AUC": auc,
            "5-Fold CV F1 (%)": f"{np.mean(cv_f1s):.2f} ± {np.std(cv_f1s):.2f}"
        })

    df_ablation = pd.DataFrame(ablation_results)
    print("\n" + "=" * 90)
    print(" [ABLATION RESULTS] FEATURE SELECTION PERFORMANCE BY REDUCTION STAGE")
    print("=" * 90)
    print(df_ablation.to_string(index=False))

    # Save Ablation CSV
    csv_path = os.path.join("outputs", "tri_stage_feature_selection_ablation.csv")
    df_ablation.to_csv(csv_path, index=False)
    print(f"\n[Saved CSV] Ablation table saved to: {csv_path}")

    # Generate Chart 1: Feature Ablation Comparison Chart
    plt.figure(figsize=(12, 6), dpi=300)
    plot_df = pd.melt(
        df_ablation,
        id_vars=["Feature Selection Level"],
        value_vars=["Test Accuracy (%)", "Recall / Sensitivity (%)", "F1-Score (%)"],
        var_name="Clinical Metric",
        value_name="Percentage (%)"
    )
    sns.barplot(data=plot_df, x="Feature Selection Level", y="Percentage (%)", hue="Clinical Metric", palette="mako")
    plt.title("Step 3: Tri-Stage Hybrid Feature Selection (Tri-HFS) Ablation Study", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Score (%)", fontsize=11)
    plt.xlabel("Feature Selection Stage", fontsize=11)
    plt.ylim(75, 100)
    plt.xticks(rotation=15, ha='right')
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart_path = os.path.join("outputs", "feature_selection_ablation_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Ablation chart saved to: {chart_path}")

    # Generate Chart 2: Top Features Stability Frequency
    stability_df = fs_meta["stage3_stability"].head(16)
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(
        data=stability_df,
        y="Feature",
        x="Selection_Frequency",
        palette="crest"
    )
    plt.title("Stage 3: ElasticNet Stability Selection Frequency (Top 16 Biomarkers across 50 Subsamples)", fontsize=11, fontweight="bold")
    plt.xlabel("Bootstrap Selection Frequency (0.0 to 1.0)", fontsize=10)
    plt.ylabel("Biomarker", fontsize=10)
    plt.xlim(0, 1.05)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    stability_chart_path = os.path.join("outputs", "tri_stage_feature_ranking_chart.png")
    plt.savefig(stability_chart_path, dpi=300)
    plt.close()
    print(f"[Saved Chart] Feature ranking chart saved to: {stability_chart_path}")

    print("\n" + "=" * 80)
    print(" Step 3 Tri-HFS Benchmark & Ablation Study Successfully Finished!")
    print("=" * 80)
    return df_ablation, final_16_feats

if __name__ == "__main__":
    run_step3_benchmark()
