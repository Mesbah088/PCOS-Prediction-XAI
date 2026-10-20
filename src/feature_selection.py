import pandas as pd
import numpy as np
import re
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from typing import List, Tuple, Dict, Any, Optional

# The 16 canonical features identified in earlier baseline literature
PAPER_16_FEATURES = [
    "Follicle No. (R)",
    "Follicle No. (L)",
    "hair growth(Y/N)",
    "Cycle(R/I)",
    "Weight gain(Y/N)",
    "Skin darkening (Y/N)",
    "Cycle length(days)",
    "Pimples(Y/N)",
    "Avg. F size (L) (mm)",
    "Marraige Status (Yrs)",
    "Avg. F size (R) (mm)",
    "RR (breaths/min)",
    "Fast food (Y/N)",
    "Weight (Kg)",
    "FSH(mIU/mL)",
    "AMH(ng/mL)"
]

def stage1_hybrid_filter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_k: int = 30
) -> Tuple[List[str], pd.DataFrame]:
    """
    Stage 1: Hybrid Filter combining Chi-Square (for categorical/discrete)
    and ANOVA F-value (for continuous clinical variables).
    """
    # MinMax non-negativity check
    X_nonneg = X_train.clip(lower=0)

    # 1. Chi-Square scores
    chi2_scores, chi2_pvals = chi2(X_nonneg, y_train)
    # 2. ANOVA F-classif scores
    f_scores, f_pvals = f_classif(X_nonneg, y_train)

    # Normalize scores to [0, 1] range to combine them fairly
    def norm_scores(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx == mn:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn + 1e-12)

    norm_chi2 = norm_scores(np.nan_to_num(chi2_scores))
    norm_f = norm_scores(np.nan_to_num(f_scores))
    combined_scores = 0.5 * norm_chi2 + 0.5 * norm_f

    filter_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Chi2_Score": chi2_scores,
        "ANOVA_F_Score": f_scores,
        "Hybrid_Score": combined_scores
    }).sort_values(by="Hybrid_Score", ascending=False)

    top_features = list(filter_df.head(min(target_k, len(X_train.columns)))["Feature"])
    print(f"[Tri-HFS Stage 1] Hybrid Filter (Chi2 + ANOVA): Reduced from {X_train.shape[1]} to {len(top_features)} features.")
    return top_features, filter_df


def stage2_wrapper_selection(
    X_train_s1: pd.DataFrame,
    y_train: pd.Series,
    target_k: int = 18,
    wrapper_type: str = "catboost",
    random_state: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Stage 2: Non-Linear Wrapper selection using CatBoost / XGBoost RFE
    to capture high-order clinical interaction terms.
    """
    wrapper_lower = wrapper_type.lower()
    if "catboost" in wrapper_lower:
        estimator = CatBoostClassifier(iterations=120, learning_rate=0.05, verbose=0, random_seed=random_state)
    elif "xgboost" in wrapper_lower:
        estimator = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss", random_state=random_state)
    elif "rf" in wrapper_lower:
        estimator = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    else:
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)

    rfe = RFE(estimator=estimator, n_features_to_select=min(target_k, X_train_s1.shape[1]), step=1)
    rfe.fit(X_train_s1, y_train)

    stage2_features = list(X_train_s1.columns[rfe.support_])
    ranking_df = pd.DataFrame({
        "Feature": X_train_s1.columns,
        "RFE_Rank": rfe.ranking_,
        "Selected": rfe.support_
    }).sort_values(by="RFE_Rank")

    print(f"[Tri-HFS Stage 2] Non-Linear Wrapper ({wrapper_type.upper()} RFE): Reduced from {X_train_s1.shape[1]} to {len(stage2_features)} features.")
    return stage2_features, ranking_df


def stage3_elasticnet_stability_selection(
    X_train_s2: pd.DataFrame,
    y_train: pd.Series,
    target_k: int = 16,
    n_bootstraps: int = 50,
    random_state: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Stage 3: Embedded Regularization & Stability Selection via ElasticNet / L1-L2 Logistic Regression.
    Runs multiple subsamples to identify core invariant biomarkers.
    """
    feature_counts = {c: 0 for c in X_train_s2.columns}
    coef_sums = {c: 0.0 for c in X_train_s2.columns}
    rng = np.random.RandomState(random_state)

    for i in range(n_bootstraps):
        # 85% subsample with replacement
        sample_indices = rng.choice(len(X_train_s2), size=int(0.85 * len(X_train_s2)), replace=True)
        sub_X = X_train_s2.iloc[sample_indices]
        sub_y = y_train.iloc[sample_indices]

        # ElasticNet / L1-penalized SAGA logistic regression
        clf = LogisticRegression(
            penalty='elasticnet',
            l1_ratio=0.5,
            C=0.8,
            solver='saga',
            max_iter=1500,
            random_state=random_state + i
        )
        try:
            clf.fit(sub_X, sub_y)
            coefs = np.abs(clf.coef_[0])
            for idx, c in enumerate(X_train_s2.columns):
                coef_val = coefs[idx]
                if coef_val > 1e-4:
                    feature_counts[c] += 1
                coef_sums[c] += coef_val
        except Exception:
            continue

    stability_df = pd.DataFrame({
        "Feature": list(feature_counts.keys()),
        "Selection_Frequency": [feature_counts[c] / n_bootstraps for c in feature_counts],
        "Mean_Absolute_Coefficient": [coef_sums[c] / n_bootstraps for c in coef_sums]
    }).sort_values(by=["Selection_Frequency", "Mean_Absolute_Coefficient"], ascending=False)

    final_k_features = list(stability_df.head(min(target_k, len(X_train_s2.columns)))["Feature"])
    print(f"[Tri-HFS Stage 3] ElasticNet Stability Selection: Isolated top {len(final_k_features)} invariant biomarkers.")
    return final_k_features, stability_df


def tri_stage_hybrid_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    stage1_k: int = 30,
    stage2_k: int = 18,
    stage3_k: int = 16,
    wrapper_type: str = "catboost",
    random_state: int = 42
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Complete Tri-Stage Hybrid Feature Selection (Tri-HFS) Workflow:
    Raw (66) -> Stage 1 Filter (30) -> Stage 2 Wrapper (18) -> Stage 3 Stability Selection (16).
    """
    print("=" * 70)
    print("  EXECUTING TRI-STAGE HYBRID FEATURE SELECTION (Tri-HFS)")
    print(f"  Input Pool: {X_train.shape[1]} features | Reduction: {X_train.shape[1]} -> {stage1_k} -> {stage2_k} -> {stage3_k}")
    print("=" * 70)

    # Stage 1: Hybrid Filter (Chi2 + ANOVA)
    s1_features, s1_df = stage1_hybrid_filter(X_train, y_train, target_k=stage1_k)
    X_train_s1 = X_train[s1_features]

    # Stage 2: Non-Linear Wrapper (CatBoost RFE)
    s2_features, s2_df = stage2_wrapper_selection(
        X_train_s1, y_train, target_k=stage2_k, wrapper_type=wrapper_type, random_state=random_state
    )
    X_train_s2 = X_train[s2_features]

    # Stage 3: Regularized ElasticNet Stability Selection
    s3_features, s3_df = stage3_elasticnet_stability_selection(
        X_train_s2, y_train, target_k=stage3_k, n_bootstraps=50, random_state=random_state
    )

    metadata = {
        "stage1_features": s1_features,
        "stage1_scores": s1_df,
        "stage2_features": s2_features,
        "stage2_ranks": s2_df,
        "stage3_features": s3_features,
        "stage3_stability": s3_df
    }

    print("\n[Consensus Complete] Top Invariant Core Biomarkers Selected:")
    for rank, feat in enumerate(s3_features, 1):
        print(f"  {rank:2d}. {feat}")
    print("=" * 70 + "\n")

    return s3_features, metadata


def two_stage_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    stage1_k: int = 30,
    stage2_k: int = 16,
    wrapper_type: str = "catboost",
    random_state: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """Backwards-compatible two-stage wrapper call."""
    top_s1, _ = stage1_hybrid_filter(X_train, y_train, target_k=stage1_k)
    top_s2, ranking_df = stage2_wrapper_selection(
        X_train[top_s1], y_train, target_k=stage2_k, wrapper_type=wrapper_type, random_state=random_state
    )
    return top_s2, ranking_df

