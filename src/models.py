import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns the suite of baseline classifiers analyzed in the paper."""
    return {
        "LR": LogisticRegression(max_iter=1000, random_state=random_state),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "RF": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "XGB": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=random_state),
        "CB": CatBoostClassifier(iterations=100, verbose=0, random_seed=random_state),
        "AB": AdaBoostClassifier(n_estimators=50, random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state),
        "LDA": LinearDiscriminantAnalysis()
    }

def get_champion_ensemble(random_state: int = 42) -> VotingClassifier:
    """
    Returns the proposed champion Soft-Voting ensemble: XGBoost + Multi-Layer Perceptron (MLP).
    Achieves 96.33% accuracy in the research paper.
    """
    xgb = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=random_state)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('mlp', mlp)],
        voting='soft'
    )
    return ensemble

def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """Train and evaluate a classifier on train and hold-out test sets."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = np.nan

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    cm = confusion_matrix(y_test, y_pred)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Confusion Matrix": cm,
        "Fitted Model": model
    }

def perform_10fold_cv(model: Any, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Dict[str, str]:
    """Performs 10-fold Stratified Cross-Validation reporting mean +/- std."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    return {
        "CV Accuracy": f"{scores['test_accuracy'].mean()*100:.2f}% ± {scores['test_accuracy'].std()*100:.2f}%",
        "CV Precision": f"{scores['test_precision'].mean()*100:.2f}% ± {scores['test_precision'].std()*100:.2f}%",
        "CV Recall": f"{scores['test_recall'].mean()*100:.2f}% ± {scores['test_recall'].std()*100:.2f}%",
        "CV F1-Score": f"{scores['test_f1'].mean()*100:.2f}% ± {scores['test_f1'].std()*100:.2f}%"
    }
