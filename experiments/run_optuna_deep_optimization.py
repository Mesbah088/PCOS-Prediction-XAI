import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import optuna
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from src.feature_engineering import engineer_clinical_features

def load_and_clean_data(file_path: str):
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    target_col = [c for c in df.columns if 'PCOS' in c][0]
    
    drop_cols = [c for c in df.columns if any(k in c.lower() for k in ['unnamed', 'sl. no', 'sl.no', 'patient file'])]
    df = df.drop(columns=drop_cols, errors='ignore')

    for col in df.columns:
        if col != target_col:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median() if not np.isnan(df[col].median()) else 0)

    return df, target_col

def optimize_pcos_pipeline():
    os.makedirs("outputs", exist_ok=True)
    print("=" * 80)
    print(" [PHASE 2] DEEP OPTUNA BAYESIAN OPTIMIZATION & THRESHOLD TUNING")
    print("=" * 80)

    data_path = os.path.join("data", "PCOS_data.csv")
    raw_df, target_col = load_and_clean_data(data_path)
    
    # 1. Domain Feature Engineering
    df_eng = engineer_clinical_features(raw_df)
    
    X = df_eng.drop(columns=[target_col])
    y = df_eng[target_col].astype(int)

    # 2. Power Transform + MinMax for handling heavy-tailed hormone distributions
    pt = PowerTransformer(method='yeo-johnson')
    X_transformed = pd.DataFrame(pt.fit_transform(X), columns=X.columns)
    
    # Scale to [0, 1] for Chi-Square non-negative requirement
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_transformed), columns=X.columns)

    # 3. 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Borderline-SMOTE on training set only
    smote = BorderlineSMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Two-Stage Feature Selection
    chi2_sel = SelectKBest(score_func=chi2, k=min(30, X_train_res.shape[1]))
    chi2_sel.fit(X_train_res, y_train_res)
    feat_stage1 = list(X_train_res.columns[chi2_sel.get_support()])

    cb_rfe = CatBoostClassifier(iterations=100, verbose=0, random_seed=42)
    rfe = RFE(estimator=cb_rfe, n_features_to_select=16, step=1)
    rfe.fit(X_train_res[feat_stage1], y_train_res)
    top16_features = list(X_train_res[feat_stage1].columns[rfe.support_])

    print(f"\nTop 16 Optimized Features: {top16_features}")

    X_tr_final = X_train_res[top16_features]
    X_te_final = X_test[top16_features]

    # 6. Optuna Hyperparameter Optimization for Base Models
    print("\n>>> Tuning LightGBM & CatBoost via Optuna (50 trials each)...")
    
    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        clf = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probs = cross_val_predict(clf, X_tr_final, y_train_res, cv=cv, method='predict_proba')[:, 1]
        return roc_auc_score(y_train_res, probs)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=40)
    best_lgb_params = study_lgb.best_params
    best_lgb_params['random_state'] = 42
    best_lgb_params['verbose'] = -1
    opt_lgb = LGBMClassifier(**best_lgb_params)

    def cb_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'depth': trial.suggest_int('depth', 3, 7),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_seed': 42,
            'verbose': 0
        }
        clf = CatBoostClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probs = cross_val_predict(clf, X_tr_final, y_train_res, cv=cv, method='predict_proba')[:, 1]
        return roc_auc_score(y_train_res, probs)

    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(cb_objective, n_trials=30)
    best_cb_params = study_cb.best_params
    best_cb_params['random_seed'] = 42
    best_cb_params['verbose'] = 0
    opt_cb = CatBoostClassifier(**best_cb_params)

    opt_xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    opt_et = ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_split=3, random_state=42)
    opt_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=0.01, max_iter=600, random_state=42)

    # 7. Champion Super-Ensemble with Out-Of-Fold Stacking
    print("\n>>> Fitting Master Stacking Ensemble (Opt-LGB + Opt-CB + Opt-XGB + Opt-ET + MLP -> ElasticNet)...")
    base_clfs = [
        ('lgb', opt_lgb),
        ('cb', opt_cb),
        ('xgb', opt_xgb),
        ('et', opt_et),
        ('mlp', opt_mlp)
    ]
    
    meta_clf = LogisticRegression(C=0.8, penalty='l2', random_state=42)
    master_stack = StackingClassifier(
        estimators=base_clfs,
        final_estimator=meta_clf,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    master_stack.fit(X_tr_final, y_train_res)

    # 8. Optimal Decision Threshold Tuning via Youden's J Index
    # Find best threshold on out-of-fold train probabilities to avoid default 0.50 bias
    oof_train_probs = cross_val_predict(master_stack, X_tr_final, y_train_res, cv=5, method='predict_proba')[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train_res, oof_train_probs)
    youden_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_j)]
    print(f"Optimal Youden's J Decision Threshold: {best_threshold:.4f}")

    # Evaluate on Test Set
    test_probs = master_stack.predict_proba(X_te_final)[:, 1]
    y_pred_opt = (test_probs >= best_threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_opt) * 100
    prec = precision_score(y_test, y_pred_opt) * 100
    rec = recall_score(y_test, y_pred_opt) * 100
    f1 = f1_score(y_test, y_pred_opt) * 100
    auc = roc_auc_score(y_test, test_probs)
    cm = confusion_matrix(y_test, y_pred_opt)

    print("\n" + "=" * 70)
    print(" 🚀 OPTIMIZED CHAMPION MODEL PERFORMANCE ON TEST SET")
    print("=" * 70)
    print(f" - Test Accuracy:  {acc:.2f}% (vs Paper Baseline: 96.33%)")
    print(f" - Precision:      {prec:.2f}% (vs Paper: 96.35%)")
    print(f" - Recall:         {rec:.2f}% (vs Paper: 96.33%)")
    print(f" - F1-Score:       {f1:.2f}% (vs Paper: 96.30%)")
    print(f" - ROC-AUC:        {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    # 9. 10-Fold Stratified Cross-Validation on the Whole Dataset
    print("\n" + "=" * 70)
    print(" 10-FOLD STRATIFIED CROSS-VALIDATION (WHOLE DATASET)")
    print("=" * 70)
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accs, cv_f1s, cv_aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv10.split(X_scaled[top16_features], y)):
        X_fold_tr, y_fold_tr = X_scaled[top16_features].iloc[train_idx], y.iloc[train_idx]
        X_fold_val, y_fold_val = X_scaled[top16_features].iloc[val_idx], y.iloc[val_idx]

        # Resample train only
        X_fold_res, y_fold_res = smote.fit_resample(X_fold_tr, y_fold_tr)
        
        m_fold = VotingClassifier(
            estimators=[('lgb', opt_lgb), ('cb', opt_cb), ('xgb', opt_xgb), ('mlp', opt_mlp)],
            voting='soft',
            weights=[3, 3, 2, 2]
        )
        m_fold.fit(X_fold_res, y_fold_res)
        
        p_val = m_fold.predict(X_fold_val)
        pr_val = m_fold.predict_proba(X_fold_val)[:, 1]

        cv_accs.append(accuracy_score(y_fold_val, p_val))
        cv_f1s.append(f1_score(y_fold_val, p_val))
        cv_aucs.append(roc_auc_score(y_fold_val, pr_val))

    print(f"-> 10-Fold Mean Accuracy: {np.mean(cv_accs)*100:.2f}% ± {np.std(cv_accs)*100:.2f}% (vs Paper: 92.04% ± 2.58%)")
    print(f"-> 10-Fold Mean F1-Score: {np.mean(cv_f1s)*100:.2f}% ± {np.std(cv_f1s)*100:.2f}% (vs Paper: 92.03% ± 2.59%)")
    print(f"-> 10-Fold Mean ROC-AUC:  {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

    # 10. Generate State-of-the-Art XAI Plots (SHAP & LIME)
    print("\n>>> Generating High-Resolution XAI Visualizations...")
    
    # Fit individual best tree model for SHAP
    opt_lgb.fit(X_tr_final, y_train_res)
    explainer = shap.TreeExplainer(opt_lgb)
    shap_vals = explainer.shap_values(X_te_final)
    
    # Beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_te_final, show=False)
    plt.title("SHAP Global Feature Importance (Optimized Champion Model)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    shap_path = os.path.join("outputs", "optimized_shap_summary.png")
    plt.savefig(shap_path, dpi=300)
    plt.close()
    print(f"Saved SHAP plot to: {shap_path}")

    # LIME Local Explainer
    print(">>> Generating LIME local explanation...")
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_tr_final),
        feature_names=list(X_tr_final.columns),
        class_names=['Non-PCOS', 'PCOS'],
        mode='classification',
        random_state=42
    )
    predict_fn_df = lambda x: master_stack.predict_proba(pd.DataFrame(x, columns=top16_features))
    exp = lime_exp.explain_instance(
        data_row=X_te_final.iloc[0].values,
        predict_fn=predict_fn_df,
        num_features=10
    )
    lime_path = os.path.join("outputs", "optimized_lime_patient_0.png")
    fig = exp.as_pyplot_figure()
    plt.title("LIME Local Patient Prediction Attribution", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(lime_path, dpi=300)
    plt.close()
    print(f"Saved LIME explanation to: {lime_path}")

    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-PCOS', 'PCOS'], yticklabels=['Non-PCOS', 'PCOS'])
    plt.title("Champion Model Confusion Matrix", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join("outputs", "optimized_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to: {cm_path}")

    # Save summary report
    metrics_summary = {
        "Model Architecture": "Optuna-Tuned Stacking (LGBM + CatBoost + XGBoost + ET + MLP -> ElasticNet)",
        "Test Accuracy (%)": f"{acc:.2f}%",
        "Test Precision (%)": f"{prec:.2f}%",
        "Test Recall (%)": f"{rec:.2f}%",
        "Test F1-Score (%)": f"{f1:.2f}%",
        "Test ROC-AUC": f"{auc:.4f}",
        "10-Fold CV Accuracy": f"{np.mean(cv_accs)*100:.2f}% ± {np.std(cv_accs)*100:.2f}%",
        "10-Fold CV F1": f"{np.mean(cv_f1s)*100:.2f}% ± {np.std(cv_f1s)*100:.2f}%",
        "10-Fold CV AUC": f"{np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}",
        "Paper Baseline 10-Fold CV": "92.04% ± 2.58%"
    }
    pd.DataFrame([metrics_summary]).to_csv("outputs/champion_model_metrics.csv", index=False)
    print("Saved champion metrics to: outputs/champion_model_metrics.csv")

    print("\n" + "=" * 80)
    print(" DEEP OPTIMIZATION EXPERIMENT COMPLETED SUCCESSFULLY ")
    print("=" * 80)

if __name__ == "__main__":
    optimize_pcos_pipeline()
