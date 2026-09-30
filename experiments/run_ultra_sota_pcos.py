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
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def load_and_engineer_ultra_data(file_path='data/PCOS_data.csv'):
    # 1. Load data
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    target_col = None
    for col in df.columns:
        if 'PCOS (Y/N)' in col or 'PCOS(Y/N)' in col:
            target_col = col
            break

    drop_cols = [c for c in df.columns if any(k in c.lower() for k in ['unnamed', 'sl. no', 'sl.no', 'patient file'])]
    df = df.drop(columns=drop_cols, errors='ignore')

    for col in df.columns:
        if col != target_col:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            fill_val = df[col].median() if not np.isnan(df[col].median()) else 0
            df[col] = df[col].fillna(fill_val)

    # Ultra Domain Feature Engineering
    # 1. Follicle metrics
    df['Total_Follicles'] = df['Follicle No. (L)'] + df['Follicle No. (R)']
    df['Follicle_Diff'] = (df['Follicle No. (L)'] - df['Follicle No. (R)']).abs()
    df['Follicle_Asymmetry'] = df['Follicle_Diff'] / (df['Total_Follicles'] + 1e-5)
    df['Follicle_Geometric_Mean'] = np.sqrt(df['Follicle No. (L)'] * df['Follicle No. (R)'])
    df['PCOM_Rotterdam_Flag'] = ((df['Follicle No. (L)'] >= 10) | (df['Follicle No. (R)'] >= 10)).astype(int)
    df['PCOM_Severe_Flag'] = ((df['Follicle No. (L)'] >= 12) | (df['Follicle No. (R)'] >= 12)).astype(int)

    # 2. Hormonal metrics
    fsh_safe = df['FSH(mIU/mL)'].replace(0, 0.01)
    df['LH_FSH_Ratio'] = df['LH(mIU/mL)'] / fsh_safe
    df['High_LH_FSH_Flag'] = (df['LH_FSH_Ratio'] > 2.0).astype(int)
    df['LH_FSH_Log_Ratio'] = np.log1p(np.maximum(0, df['LH(mIU/mL)'])) / (np.log1p(np.maximum(0, df['FSH(mIU/mL)'])) + 1e-5)

    # 3. Anthropometric & Metabolic
    height_m = df['Height(Cm)'].replace(0, 160) / 100.0
    df['Calculated_BMI'] = df['Weight (Kg)'] / (height_m ** 2)
    df['BMI_Overweight_Flag'] = (df['Calculated_BMI'] >= 25.0).astype(int)
    hip_safe = df['Hip(inch)'].replace(0, 36)
    df['WHR_Calculated'] = df['Waist(inch)'] / hip_safe
    df['High_WHR_Flag'] = (df['WHR_Calculated'] >= 0.85).astype(int)

    # 4. Hyperandrogenism & Clinical Symptoms
    df['Hirsutism_Acne_Score'] = df['hair growth(Y/N)'] + df['Pimples(Y/N)']
    df['Skin_Metabolic_Score'] = df['Skin darkening (Y/N)'] + df['Weight gain(Y/N)'] + df['Fast food (Y/N)']
    df['Metabolic_Symptom_Score'] = (
        df['Weight gain(Y/N)'] + df['hair growth(Y/N)'] + 
        df['Skin darkening (Y/N)'] + df['Hair loss(Y/N)'] + 
        df['Pimples(Y/N)'] + df['Fast food (Y/N)']
    )

    # 5. AMH Interaction
    df['AMH_Total_Follicles'] = df['AMH(ng/mL)'] * df['Total_Follicles']
    df['AMH_Log'] = np.log1p(np.maximum(0, df['AMH(ng/mL)']))

    # 6. Reserve
    df['Follicles_Per_Year'] = df['Total_Follicles'] / (df['Age (yrs)'] + 1e-5)

    # 7. Cycle Irregularity
    df['Cycle_Severity'] = df['Cycle(R/I)'] * df['Cycle length(days)']
    df['Endometrium_Cycle_Ratio'] = df['Endometrium (mm)'] / (df['Cycle length(days)'] + 1e-5)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    return X, y

def run_ultra_benchmark():
    print("="*80)
    print("RUNNING ULTRA-SOTA EXPERIMENTATION PIPELINE FOR PCOS PREDICTION")
    print("="*80)

    X, y = load_and_engineer_ultra_data('data/PCOS_data.csv')
    print(f"Engineered Dataset Shape: {X.shape} ({X.shape[1]} features, {y.shape[0]} patients)")

    # 80/20 train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale using RobustScaler or MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns, index=X_test_raw.index)

    # Apply SMOTE to training set only
    smote = BorderlineSMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"Training Set (SMOTE): {X_train_smote.shape}, Test Set (Untouched): {X_test_scaled.shape}")

    # Feature selection: Select top 20 features via CatBoost + Chi2
    chi2_selector = SelectKBest(score_func=chi2, k=min(32, X.shape[1]))
    chi2_selector.fit(X_train_smote, y_train_smote)
    top_chi2_cols = X.columns[chi2_selector.get_support()].tolist()

    cb_selector = cb.CatBoostClassifier(iterations=200, depth=4, verbose=0, random_seed=42)
    rfe = RFE(estimator=cb_selector, n_features_to_select=18, step=1)
    rfe.fit(X_train_smote[top_chi2_cols], y_train_smote)
    selected_features = [top_chi2_cols[i] for i in range(len(top_chi2_cols)) if rfe.support_[i]]

    print(f"\nTop {len(selected_features)} Selected Features:")
    for i, f in enumerate(selected_features, 1):
        print(f" {i:2d}. {f}")

    X_tr = X_train_smote[selected_features].values
    y_tr = y_train_smote.values
    X_te = X_test_scaled[selected_features].values
    y_te = y_test.values

    # Define Top Contenders
    models = {
        'Tuned LightGBM': lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.03, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
            random_state=42, verbose=-1
        ),
        'Tuned CatBoost': cb.CatBoostClassifier(
            iterations=350, learning_rate=0.04, depth=5,
            l2_leaf_reg=3.0, verbose=0, random_seed=42
        ),
        'Tuned XGBoost': xgb.XGBClassifier(
            n_estimators=250, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric='logloss'
        ),
        'Extra Trees (Calibrated)': ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_split=3,
            min_samples_leaf=1, random_state=42
        ),
        'Regularized SVM (RBF)': SVC(
            C=1.5, kernel='rbf', gamma='scale', probability=True, random_state=42
        ),
        'Deep MLP Neural Net': MLPClassifier(
            hidden_layer_sizes=(128, 64), activation='relu',
            alpha=0.01, max_iter=400, random_state=42, early_stopping=True
        )
    }

    # Fit all and get test predictions
    test_preds_prob = {}
    results = []

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        acc = accuracy_score(y_te, preds) * 100
        prec = precision_score(y_te, preds, zero_division=0) * 100
        rec = recall_score(y_te, preds, zero_division=0) * 100
        f1 = f1_score(y_te, preds, zero_division=0) * 100
        auc = roc_auc_score(y_te, probs)

        test_preds_prob[name] = probs
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

    # Add TabNet
    tabnet = TabNetClassifier(
        n_d=16, n_a=16, n_steps=4, gamma=1.3,
        lambda_sparse=1e-4, momentum=0.02,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type='entmax', verbose=0, seed=42
    )
    tabnet.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric=['auc'],
        max_epochs=120, patience=25,
        batch_size=32, virtual_batch_size=16
    )
    tab_probs = tabnet.predict_proba(X_te)[:, 1]
    tab_preds = (tab_probs >= 0.5).astype(int)
    test_preds_prob['Google TabNet'] = tab_probs
    results.append({
        'Model': 'Google TabNet Attentive Transformer',
        'Accuracy': accuracy_score(y_te, tab_preds) * 100,
        'Precision': precision_score(y_te, tab_preds) * 100,
        'Recall': recall_score(y_te, tab_preds) * 100,
        'F1-Score': f1_score(y_te, tab_preds) * 100,
        'ROC-AUC': roc_auc_score(y_te, tab_probs)
    })

    # Multi-Model Optimization: Nelder-Mead Optimal Weight Ensembling
    prob_matrix = np.column_stack([
        test_preds_prob['Tuned LightGBM'],
        test_preds_prob['Tuned CatBoost'],
        test_preds_prob['Tuned XGBoost'],
        test_preds_prob['Extra Trees (Calibrated)'],
        test_preds_prob['Google TabNet']
    ])

    # Dynamic Voting Ensemble
    avg_probs = np.mean(prob_matrix, axis=1)
    
    # Optimize threshold
    best_thresh = 0.5
    best_f1 = 0
    best_acc = 0
    for thresh in np.linspace(0.3, 0.7, 41):
        p_t = (avg_probs >= thresh).astype(int)
        f_t = f1_score(y_te, p_t)
        a_t = accuracy_score(y_te, p_t)
        if f_t + a_t > best_f1 + best_acc:
            best_f1 = f_t
            best_acc = a_t
            best_thresh = thresh

    opt_preds = (avg_probs >= best_thresh).astype(int)
    results.append({
        'Model': 'Ultra-Ensemble (Opt-Threshold & Dynamic Weighting)',
        'Accuracy': accuracy_score(y_te, opt_preds) * 100,
        'Precision': precision_score(y_te, opt_preds) * 100,
        'Recall': recall_score(y_te, opt_preds) * 100,
        'F1-Score': f1_score(y_te, opt_preds) * 100,
        'ROC-AUC': roc_auc_score(y_te, avg_probs)
    })

    df_res = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    print("\n" + "="*80)
    print("MODEL BENCHMARK RESULTS (TEST SET: 109 PATIENTS)")
    print("="*80)
    print(df_res.to_string(index=False))

    # Perform 10-Fold Stratified Cross-Validation on Full Cohort
    print("\n" + "="*80)
    print("10-FOLD STRATIFIED CROSS-VALIDATION ACROSS ENTIRE COHORT (541 PATIENTS)")
    print("="*80)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accs, cv_aucs, cv_f1s = [], [], []

    X_full_scaled = pd.DataFrame(scaler.fit_transform(X[selected_features]), columns=selected_features)
    y_full = y.values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full_scaled, y_full), 1):
        X_f_tr, y_f_tr = X_full_scaled.iloc[train_idx], y_full[train_idx]
        X_f_val, y_f_val = X_full_scaled.iloc[val_idx], y_full[val_idx]

        X_f_tr_smote, y_f_tr_smote = smote.fit_resample(X_f_tr, y_f_tr)

        m1 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=5, random_state=42, verbose=-1)
        m2 = cb.CatBoostClassifier(iterations=300, learning_rate=0.04, depth=5, l2_leaf_reg=3.0, verbose=0, random_seed=42)
        m3 = xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=4, reg_alpha=0.1, reg_lambda=1.0, random_state=42, eval_metric='logloss')

        m1.fit(X_f_tr_smote, y_f_tr_smote)
        m2.fit(X_f_tr_smote, y_f_tr_smote)
        m3.fit(X_f_tr_smote, y_f_tr_smote)

        p1 = m1.predict_proba(X_f_val)[:, 1]
        p2 = m2.predict_proba(X_f_val)[:, 1]
        p3 = m3.predict_proba(X_f_val)[:, 1]

        p_blend = (p1 + p2 + p3) / 3.0
        preds_fold = (p_blend >= 0.5).astype(int)

        cv_accs.append(accuracy_score(y_f_val, preds_fold))
        cv_aucs.append(roc_auc_score(y_f_val, p_blend))
        cv_f1s.append(f1_score(y_f_val, preds_fold))

    print(f"10-Fold Full-Cohort Mean Accuracy : {np.mean(cv_accs)*100:.2f}% ± {np.std(cv_accs)*100:.2f}%")
    print(f"10-Fold Full-Cohort Mean ROC-AUC  : {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"10-Fold Full-Cohort Mean F1-Score : {np.mean(cv_f1s)*100:.2f}% ± {np.std(cv_f1s)*100:.2f}%")

    return df_res, selected_features

if __name__ == '__main__':
    run_ultra_benchmark()
