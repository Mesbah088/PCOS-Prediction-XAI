import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)
from scipy.stats import wilcoxon

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_comprehensive_base_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns 12 diverse base machine learning classifiers spanning:
    - Tree Gradient Boosters (LightGBM, CatBoost, XGBoost, GradientBoosting)
    - Tree Ensembles (Random Forest, Extra Trees, AdaBoost)
    - Neural Architectures (Multi-Layer Perceptron - MLP)
    - Kernel & Classical Classifiers (SVM RBF, Logistic Regression, LDA, KNN, Gaussian Naive Bayes)
    """
    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=14,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=150,
            learning_rate=0.04,
            depth=4,
            random_seed=random_state,
            verbose=0
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.01,
            max_iter=600,
            random_state=random_state
        ),
        "SVM (RBF Kernel)": SVC(
            C=1.5,
            kernel='rbf',
            probability=True,
            random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ),
        "Linear Discriminant (LDA)": LinearDiscriminantAnalysis(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        "Gaussian Naive Bayes": GaussianNB()
    }


def get_champion_soft_voting_ensemble(random_state: int = 42) -> VotingClassifier:
    """
    Constructs the Champion Soft-Voting Super-Ensemble:
    Fuses diverse gradient tree boosters (LightGBM + CatBoost + XGBoost) with
    continuous neural probability densities from a tuned Multi-Layer Perceptron (MLP).
    """
    lgbm = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=14, random_state=random_state, verbose=-1)
    cat = CatBoostClassifier(iterations=150, learning_rate=0.04, depth=4, random_seed=random_state, verbose=0)
    xgb = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, eval_metric="logloss", random_state=random_state)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01, max_iter=600, random_state=random_state)

    ensemble = VotingClassifier(
        estimators=[
            ('lgbm', lgbm),
            ('cat', cat),
            ('xgb', xgb),
            ('mlp', mlp)
        ],
        voting='soft',
        weights=[1.2, 1.2, 1.0, 0.8]
    )
    return ensemble


def get_champion_stacking_ensemble(random_state: int = 42) -> StackingClassifier:
    """
    Constructs a two-tier Stacking Meta-Learner:
    Level-0 estimators: LightGBM, CatBoost, XGBoost, and MLP.
    Level-1 meta-estimator: L2-Regularized Logistic Regression with cross-validated out-of-fold predictions.
    """
    lgbm = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, num_leaves=14, random_state=random_state, verbose=-1)
    cat = CatBoostClassifier(iterations=150, learning_rate=0.04, depth=4, random_seed=random_state, verbose=0)
    xgb = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, eval_metric="logloss", random_state=random_state)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01, max_iter=600, random_state=random_state)

    stacking = StackingClassifier(
        estimators=[
            ('lgbm', lgbm),
            ('cat', cat),
            ('xgb', xgb),
            ('mlp', mlp)
        ],
        final_estimator=LogisticRegression(C=0.5, penalty='l2', max_iter=1000, random_state=random_state),
        cv=3,
        n_jobs=-1,
        passthrough=False
    )
    return stacking


def get_calibrated_champion(base_model: Any, method: str = 'sigmoid') -> CalibratedClassifierCV:
    """
    Applies post-hoc Platt Scaling (sigmoid) or Isotonic Regression calibration
    to ensure diagnostic confidence probabilities reflect true disease risk.
    """
    return CalibratedClassifierCV(estimator=base_model, method=method, cv=5, n_jobs=-1)


def evaluate_clinical_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Comprehensive clinical evaluation computing:
    Accuracy, Recall/Sensitivity (safety), Specificity, Precision, F1-Score, ROC-AUC, Brier Score, and Confusion Matrix.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
    else:
        y_proba = y_pred.astype(float)
        auc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)

    acc = accuracy_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0

    return {
        "Accuracy (%)": acc,
        "Recall (%)": rec,
        "Specificity (%)": spec,
        "Precision (%)": prec,
        "F1-Score (%)": f1,
        "ROC-AUC": auc,
        "Brier Score": brier,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "model": model
    }


from sklearn.model_selection import StratifiedKFold, cross_validate

def compute_cross_val_metrics(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs Stratified Cross-Validation on the training partition
    in a single parallel pass to compute mean ± standard deviation and fold-wise vectors for Wilcoxon testing.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_res = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=['f1', 'accuracy', 'recall'],
        n_jobs=-1
    )
    f1_scores = cv_res['test_f1']
    acc_scores = cv_res['test_accuracy']
    rec_scores = cv_res['test_recall']

    return {
        "CV_F1_Mean": f1_scores.mean() * 100,
        "CV_F1_Std": f1_scores.std() * 100,
        "CV_F1_Str": f"{f1_scores.mean()*100:.2f} ± {f1_scores.std()*100:.2f}",
        "CV_Acc_Str": f"{acc_scores.mean()*100:.2f} ± {acc_scores.std()*100:.2f}",
        "CV_Rec_Str": f"{rec_scores.mean()*100:.2f} ± {rec_scores.std()*100:.2f}",
        "fold_f1_scores": f1_scores
    }


def perform_wilcoxon_significance(
    champion_f1s: np.ndarray,
    baseline_f1s: np.ndarray
) -> Tuple[float, bool]:
    """
    Computes the Wilcoxon Signed-Rank Test between cross-validation fold scores.
    Returns (p-value, is_significant_at_0.05).
    """
    diffs = champion_f1s - baseline_f1s
    if np.all(diffs == 0):
        return 1.0, False
    try:
        stat, p_val = wilcoxon(champion_f1s, baseline_f1s, alternative='greater')
        return float(p_val), bool(p_val < 0.05)
    except Exception:
        return 1.0, False
