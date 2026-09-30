import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional

def load_and_preprocess_data(
    file_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list, MinMaxScaler]:
    """
    Load and preprocess PCOS dataset following the paper's methodology:
    1. Drop irrelevant columns ('Sl. No', 'Patient File No.', unnamed columns).
    2. Handle missing values (e.g. Marriage Status, Fast Food).
    3. MinMax scale the numeric feature variables.
    4. Perform 80/20 train-test split before oversampling.
    5. Apply SMOTE exclusively to the training set to prevent data leakage.
    """
    # 1. Load data
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, sheet_name=1 if 'without_infertility' in file_path else 0)
    else:
        df = pd.read_csv(file_path)

    # Clean column names (strip leading/trailing whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Target column determination
    target_col = None
    for col in df.columns:
        if 'PCOS (Y/N)' in col or 'PCOS(Y/N)' in col:
            target_col = col
            break

    if target_col is None:
        raise ValueError("Target column 'PCOS (Y/N)' not found in dataset.")

    # 2. Drop redundant/irrelevant identifiers
    drop_cols = [c for c in df.columns if any(k in c.lower() for k in ['unnamed', 'sl. no', 'sl.no', 'patient file'])]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Convert object columns to numeric where possible (cleaning characters like '1.99.', '?', etc.)
    for col in df.columns:
        if col != target_col:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Missing value imputation
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            fill_val = df[col].median() if not np.isnan(df[col].median()) else 0
            df[col] = df[col].fillna(fill_val)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    feature_names = list(X.columns)

    # 4. MinMax Scaling
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    # 5. Strict 80/20 Train-Test split before SMOTE to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. Apply SMOTE exclusively on training partition
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Data Loaded Successfully:")
    print(f" - Original dataset shape: {df.shape}")
    print(f" - Training samples before SMOTE: {len(X_train)} (Positive: {y_train.sum()}, Negative: {len(y_train) - y_train.sum()})")
    print(f" - Training samples after SMOTE: {len(X_train_res)} (Positive: {y_train_res.sum()}, Negative: {len(y_train_res) - y_train_res.sum()})")
    print(f" - Untouched Test samples: {len(X_test)} (Positive: {y_test.sum()}, Negative: {len(y_test) - y_test.sum()})")

    return X_train_res, X_test, y_train_res, y_test, feature_names, scaler
