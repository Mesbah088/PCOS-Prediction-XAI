import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from typing import List, Tuple, Dict

# The 16 canonical features identified by CatBoost RFE in the paper (Table 4)
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

def two_stage_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    stage1_k: int = 30,
    stage2_k: int = 16,
    wrapper_type: str = "catboost",
    random_state: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Two-Stage Hybrid Feature Selection Framework:
    - Stage 1: Chi-Square (chi2) statistical filter to select top `stage1_k` features.
    - Stage 2: Recursive Feature Elimination (RFE) using wrapper model to isolate top `stage2_k` features.
    """
    # ------------------ STAGE 1: Chi-Square Filter ------------------
    # Features must be non-negative (which is satisfied by MinMax scaling [0, 1])
    selector_chi2 = SelectKBest(score_func=chi2, k=min(stage1_k, X_train.shape[1]))
    selector_chi2.fit(X_train, y_train)

    stage1_mask = selector_chi2.get_support()
    stage1_features = list(X_train.columns[stage1_mask])
    X_train_stage1 = X_train[stage1_features]

    print(f"\n[Stage 1] Chi-Square Filter: Retained {len(stage1_features)} features.")

    # ------------------ STAGE 2: RFE Wrapper Selection ------------------
    wrapper_type = wrapper_type.lower()
    if wrapper_type == "catboost":
        estimator = CatBoostClassifier(iterations=100, verbose=0, random_seed=random_state)
    elif wrapper_type == "xgboost":
        estimator = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=random_state)
    elif wrapper_type == "rf":
        estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif wrapper_type == "lr":
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unsupported wrapper: {wrapper_type}. Choose from: catboost, xgboost, rf, lr")

    rfe = RFE(estimator=estimator, n_features_to_select=stage2_k, step=1)
    rfe.fit(X_train_stage1, y_train)

    final_features = list(X_train_stage1.columns[rfe.support_])
    ranking = pd.DataFrame({
        "Feature": stage1_features,
        "Ranking": rfe.ranking_,
        "Selected": rfe.support_
    }).sort_values(by="Ranking")

    print(f"[Stage 2] RFE ({wrapper_type.upper()}): Extracted final {len(final_features)} features.")
    return final_features, ranking
