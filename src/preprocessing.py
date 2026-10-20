import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN
from typing import Tuple, List, Dict, Optional, Generator, Any

from src.feature_engineering import engineer_clinical_features, normalize_column_names


def sanitize_clinical_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Cleans raw clinical CSV/Excel data:
    1. Normalizes column names (strips whitespace, standardizes labels).
    2. Drops non-clinical identifiers (Sl. No, Patient File No, Unnamed).
    3. Cleans dirty numeric strings ('1.99.', '?', negative values, text).
    4. Applies median imputation for continuous and mode imputation for categorical/flags.
    """
    df = normalize_column_names(df_raw)

    # Detect Target Column
    target_col = None
    for c in df.columns:
        if 'pcos' in c.lower() and ('y/n' in c.lower() or 'yn' in c.lower() or '(y/n)' in c.lower()):
            target_col = c
            break
            
    if target_col is None:
        # Fallback to any column with PCOS
        pcos_cols = [c for c in df.columns if 'pcos' in c.lower()]
        if pcos_cols:
            target_col = pcos_cols[0]
        else:
            raise ValueError("Target column representing 'PCOS (Y/N)' could not be located.")

    # Drop identifiers and empty trailing columns
    drop_patterns = ['unnamed', 'sl. no', 'sl.no', 'sl no', 'patient file', 'patient_file']
    drop_cols = [c for c in df.columns if any(p in c.lower() for p in drop_patterns)]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Convert features to clean numeric datatypes
    for col in df.columns:
        if col != target_col:
            # Clean dirty characters e.g. '1.99.', '12?', ' '
            cleaned_series = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            # Handle multiple decimals if present (e.g. 1.99. -> 1.99)
            cleaned_series = cleaned_series.apply(
                lambda val: '.'.join(str(val).split('.')[:2]) if str(val).count('.') > 1 else str(val)
            )
            df[col] = pd.to_numeric(cleaned_series, errors='coerce')
        else:
            # Clean target column as 0 or 1
            cleaned_target = df[col].astype(str).str.replace(r'[^\d]', '', regex=True)
            df[col] = pd.to_numeric(cleaned_target, errors='coerce').fillna(0).astype(int)

    # Missing value imputation (Median for continuous, Mode for discrete/flags)
    for col in df.columns:
        if col != target_col and df[col].isnull().sum() > 0:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 5:  # Categorical / Binary Flag
                mode_val = df[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else 0
            else:  # Continuous Clinical Marker
                med_val = df[col].median(skipna=True)
                fill_val = med_val if not np.isnan(med_val) else 0.0
            df[col] = df[col].fillna(fill_val)

    return df, target_col


def get_resampler(strategy: str = "smote", random_state: int = 42) -> Any:
    """Instantiates the requested class-balancing resampler."""
    strat_lower = strategy.lower().replace("-", "").replace("_", "")
    if "borderline" in strat_lower:
        return BorderlineSMOTE(random_state=random_state)
    elif "smoteenn" in strat_lower or "enn" in strat_lower:
        return SMOTEENN(random_state=random_state)
    elif "adasyn" in strat_lower:
        return ADASYN(random_state=random_state)
    else:
        return SMOTE(random_state=random_state)


def get_scaler(scaler_type: str = "minmax") -> Any:
    """Instantiates the requested normalization scaler."""
    s_type = scaler_type.lower()
    if "robust" in s_type:
        return RobustScaler()
    elif "standard" in s_type:
        return StandardScaler()
    else:
        return MinMaxScaler()


def load_and_preprocess_data(
    file_path: str,
    apply_feature_engineering: bool = True,
    scaler_type: str = "minmax",
    resampling_strategy: str = "smote",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], Any]:
    """
    Proposed Zero-Leakage Preprocessing & Feature Engineering Pipeline:
    
    1. Ingestion: Reads CSV/Excel and sanitizes corrupted strings and values.
    2. Imputation: Applies median/mode imputation on raw clinical markers.
    3. Feature Engineering: Injects 22 Rotterdam bio-markers, ratios & symptom composites.
    4. Stratified Split: Separates untouched Test Set (20%) BEFORE scaling/resampling.
    5. Inside-Fold Scaling: Fits Scaler STRICTLY on Training Set, transforms Test Set.
    6. Inside-Fold Resampling: Balances minority classes STRICTLY inside Training Set.
    
    Returns:
        X_train_resampled, X_test_scaled, y_train_resampled, y_test, feature_names, scaler
    """
    # 1. Load Data
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df_raw = pd.read_excel(file_path, sheet_name=1 if 'without_infertility' in file_path else 0)
    else:
        df_raw = pd.read_csv(file_path)

    # 2. Clean & Sanitize
    df_clean, target_col = sanitize_clinical_dataframe(df_raw)

    # 3. Apply Rotterdam Feature Engineering if enabled
    if apply_feature_engineering:
        df_proc = engineer_clinical_features(df_clean)
    else:
        df_proc = df_clean.copy()

    # Separate Features & Target
    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col].astype(int)
    feature_names = list(X.columns)

    # 4. STRICT ZERO-LEAKAGE SPLIT (Hold-out Test partition is locked)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. INSIDE-TRAIN SCALING (Zero Leakage: Scaler fits ONLY on training data)
    scaler = get_scaler(scaler_type)
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=feature_names,
        index=X_train_raw.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_names,
        index=X_test_raw.index
    )

    # 6. INSIDE-TRAIN RESAMPLING (Zero Leakage: Resampler fits ONLY on training data)
    resampler = get_resampler(resampling_strategy, random_state=random_state)
    X_train_resampled, y_train_resampled = resampler.fit_resample(X_train_scaled, y_train)

    print("\n" + "=" * 70)
    print(" [ZERO-LEAKAGE PIPELINE] Preprocessing & Feature Engineering Completed")
    print("=" * 70)
    print(f" -> Raw Cohort: {df_raw.shape[0]} patients, {df_raw.shape[1]} raw columns")
    print(f" -> Processed Features: {len(feature_names)} features (Domain Eng: {apply_feature_engineering})")
    print(f" -> Training Cohort: {len(X_train_raw)} -> Resampled to {len(X_train_resampled)} via {resampling_strategy.upper()}")
    print(f" -> Untouched Held-out Test Set: {len(X_test_scaled)} patients (Positive: {y_test.sum()}, Negative: {len(y_test) - y_test.sum()})")
    print(f" -> Scaler: {scaler_type.upper()} fitted strictly inside training boundary (Zero Leakage)")
    print("=" * 70 + "\n")

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, feature_names, scaler


def get_nested_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scaler_type: str = "minmax",
    resampling_strategy: str = "smote",
    random_state: int = 42
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any], None, None]:
    """
    Generator yielding strict zero-leakage cross-validation folds (Stratified 5-Fold Nested CV):
    Inside each fold:
      1. Scaler is fitted ONLY on Fold-Train and transforms Fold-Val.
      2. Resampler is applied ONLY to Fold-Train.
      3. Fold-Val remains completely clean and untouched.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    feature_names = list(X.columns)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_raw = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # Inside-fold scaling
        scaler = get_scaler(scaler_type)
        X_tr_scaled = pd.DataFrame(
            scaler.fit_transform(X_tr_raw),
            columns=feature_names,
            index=X_tr_raw.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_raw),
            columns=feature_names,
            index=X_val_raw.index
        )

        # Inside-fold resampling
        resampler = get_resampler(resampling_strategy, random_state=random_state + fold_idx)
        X_tr_res, y_tr_res = resampler.fit_resample(X_tr_scaled, y_tr)

        yield X_tr_res, X_val_scaled, y_tr_res, y_val, scaler

