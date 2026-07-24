"""
preprocess.py — Credit Risk Dataset Preprocessing Pipeline

This script reproduces the full data preprocessing pipeline for the
credit risk dataset (credit_risk_dataset.csv).

Steps
-----
1. Load raw data
2. Stratified train / test split  (80% / 20%, random_state=42)
3. Outlier removal in each split (person_age > 100, person_emp_length > 60)
4. Label encoding         (loan_grade A-G → 0-6, cb_person_default_on_file Y/N → 1/0)
5. One-hot encoding       (person_home_ownership, loan_intent), then align the
                           test feature schema to the training feature schema.
6. Missing value imputation  (statistics computed on training split only)
7. IQR clipping and Min-Max scaling (statistics computed on training split only)
8. Save train_processed.csv and test_processed.csv

Libraries used: NumPy, Pandas only — no scikit-learn.
"""

import os
import numpy as np
import pandas as pd


#  Resolve paths relative to this file for reproducible CLI runs.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH   = os.path.join(SCRIPT_DIR, "original_data", "credit_risk_dataset.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "preprocessed_data")
TRAIN_OUT  = os.path.join(OUTPUT_DIR, "train_processed.csv")
TEST_OUT   = os.path.join(OUTPUT_DIR, "test_processed.csv")

TARGET_COL   = "loan_status"
RANDOM_STATE = 42
TEST_RATIO   = 0.20

# Ordinal mapping for loan grade (A is best → 0, G is worst → 6)
GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}

# Categorical columns to one-hot encode
OHE_COLS = ["person_home_ownership", "loan_intent"]

# Numeric columns to Min-Max scale
NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]


#  Step 1 — Load raw data
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[1] Raw data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"    Missing values — person_emp_length: {df['person_emp_length'].isna().sum()}, "
          f"loan_int_rate: {df['loan_int_rate'].isna().sum()}")
    print(f"    Target distribution: {df[TARGET_COL].value_counts().to_dict()}")
    return df


#  Step 2 — Outlier removal
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records with biologically implausible values.
      - person_age > 100  (dataset contains values up to 144)
      - person_emp_length > 60  (dataset contains values up to 123)
    """
    n_before = len(df)
    df = df[df["person_age"] <= 100].copy()
    df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)].copy()
    n_removed = n_before - len(df)
    print(f"[2] Outlier removal: removed {n_removed} rows → {len(df)} rows remaining")
    return df.reset_index(drop=True)


#  Step 3 — Label encoding (before split, order-independent)
def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode ordinal / binary categorical features:
      - loan_grade: A→0, B→1, C→2, D→3, E→4, F→5, G→6
      - cb_person_default_on_file: Y→1, N→0
    """
    df = df.copy()
    df["loan_grade"] = df["loan_grade"].map(GRADE_MAP)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N": 0})
    print("[3] Label encoding applied: loan_grade, cb_person_default_on_file")
    return df


#  Step 4 — One-hot encoding
def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode nominal categorical features:
      - person_home_ownership  → home_ownership_MORTGAGE/OTHER/OWN/RENT
      - loan_intent            → loan_intent_DEBTCONSOLIDATION/EDUCATION/
                                 HOMEIMPROVEMENT/MEDICAL/PERSONAL/VENTURE
    The original column is dropped; all dummies are kept (no drop_first).
    """
    df = df.copy()
    for col in OHE_COLS:
        dummies = pd.get_dummies(df[col], prefix=col.replace("person_home_ownership", "home_ownership"),
                                 drop_first=False)
        # Convert bool columns to int
        dummies = dummies.astype(int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    print(f"[4] One-hot encoding applied: {OHE_COLS}  → {df.shape[1]} total columns")
    return df


def align_feature_columns(train_df: pd.DataFrame,
                          test_df: pd.DataFrame):
    """Align the test feature schema to the training feature schema.

    Encoding the two splits independently is safe only when their dummy columns
    are made identical afterwards. Categories not observed in training are not
    modelled; categories absent from test are represented by all-zero columns.
    The target column is preserved separately so it cannot be created or lost by
    the alignment step.
    """
    feature_columns = [col for col in train_df.columns if col != TARGET_COL]
    test_target = test_df[TARGET_COL].copy()

    train_features = train_df[feature_columns].copy()
    test_features = test_df.drop(columns=[TARGET_COL]).reindex(
        columns=feature_columns,
        fill_value=0,
    )
    train_features[TARGET_COL] = train_df[TARGET_COL].to_numpy()
    test_features[TARGET_COL] = test_target.to_numpy()

    print(
        "[5] Feature schemas aligned: "
        f"{len(feature_columns)} model features in both splits"
    )
    return train_features, test_features


def approximate_mode(class_counts, n_draws, rng):
    """NumPy implementation of sklearn.utils.extmath._approximate_mode."""
    class_counts = np.asarray(class_counts, dtype=np.int64)
    continuous = class_counts / class_counts.sum() * n_draws
    floored = np.floor(continuous).astype(np.int64)
    need_to_add = int(n_draws - floored.sum())

    if need_to_add > 0:
        remainders = continuous - floored
        values = np.sort(np.unique(remainders))[::-1]
        for value in values:
            inds = np.flatnonzero(remainders == value)
            add_now = min(len(inds), need_to_add)
            chosen = rng.choice(inds, size=add_now, replace=False)
            floored[chosen] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break

    return floored


#  Step 5 — Stratified 80/20 train / test split
def stratified_split(df: pd.DataFrame,
                     test_ratio: float = TEST_RATIO,
                     random_state: int = RANDOM_STATE):
    """
    Reproduce the original project split exactly.

    The report-compatible CSVs were generated by sklearn's stratified
    train_test_split on the raw dataset before outlier removal. This function
    mirrors that algorithm with NumPy so the demo has no sklearn dependency.
    """
    y = df[TARGET_COL].to_numpy()
    classes, y_indices = np.unique(y, return_inverse=True)
    class_counts = np.bincount(y_indices)
    rng = np.random.RandomState(random_state)

    n_samples = len(df)
    n_test = int(np.ceil(test_ratio * n_samples))
    n_train = n_samples - n_test

    class_indices = np.split(
        np.argsort(y_indices, kind="mergesort"),
        np.cumsum(class_counts)[:-1],
    )
    n_i = approximate_mode(class_counts, n_train, rng)
    t_i = approximate_mode(class_counts - n_i, n_test, rng)

    train = []
    test = []
    for i in range(len(classes)):
        permutation = rng.permutation(class_counts[i])
        permuted_class_indices = class_indices[i].take(permutation, mode="clip")
        train.extend(permuted_class_indices[:n_i[i]])
        test.extend(permuted_class_indices[n_i[i]:n_i[i] + t_i[i]])

    train_idx = rng.permutation(train)
    test_idx = rng.permutation(test)
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    print(f"[5] Stratified split (80/20, seed={random_state}):")
    print(f"    Train: {len(train_df)} rows  "
          f"(positive ratio: {train_df[TARGET_COL].mean():.4f})")
    print(f"    Test : {len(test_df)} rows  "
          f"(positive ratio: {test_df[TARGET_COL].mean():.4f})")
    return train_df.copy(), test_df.copy()


#  Step 6 — Missing value imputation
#  (computed on training set only, applied to test)
def impute_missing(train_df: pd.DataFrame,
                   test_df: pd.DataFrame):
    """
    Fill missing values using the original project's report-compatible stats.
    Columns with missing values:
      - person_emp_length  uses the training mean
      - loan_int_rate      uses the training median

    Statistics are computed on the training split only and then applied to
    both splits to prevent data leakage.
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    emp_mean = float(train_df["person_emp_length"].mean())
    rate_median = float(train_df["loan_int_rate"].median())
    train_df["person_emp_length"] = train_df["person_emp_length"].fillna(emp_mean)
    test_df["person_emp_length"] = test_df["person_emp_length"].fillna(emp_mean)
    train_df["loan_int_rate"] = train_df["loan_int_rate"].fillna(rate_median)
    test_df["loan_int_rate"] = test_df["loan_int_rate"].fillna(rate_median)

    print("[6] Missing value imputation (training-set stats):")
    print(f"    person_emp_length: filled with mean = {emp_mean:.4f}")
    print(f"    loan_int_rate: filled with median = {rate_median:.4f}")

    return train_df, test_df


def clip_person_income(train_df: pd.DataFrame,
                       test_df: pd.DataFrame):
    """
    Clip person_income with the training-set IQR upper fence.

    This is required to reproduce the original processed CSVs: income values
    above Q3 + 1.5 * IQR are capped before Min-Max scaling.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    q1 = float(train_df["person_income"].quantile(0.25))
    q3 = float(train_df["person_income"].quantile(0.75))
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    train_df["person_income"] = train_df["person_income"].clip(upper=upper)
    test_df["person_income"] = test_df["person_income"].clip(upper=upper)
    print(f"[7] person_income IQR clipping: upper = {upper:.4f}")
    return train_df, test_df, upper


#  Step 7 — Min-Max scaling
#  (computed on training set only, applied to test)
def minmax_scale(train_df: pd.DataFrame,
                 test_df: pd.DataFrame):
    """
    Apply Min-Max normalization to numeric features:

        x_scaled = (x - x_min) / (x_max - x_min)

    where x_min and x_max are derived from the training set only.
    This prevents data leakage from the test set.
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    scale_params = {}
    for col in NUMERIC_COLS:
        col_min = float(train_df[col].min())
        col_max = float(train_df[col].max())
        scale_params[col] = (col_min, col_max)
        scale = 1.0 / (col_max - col_min) if col_max != col_min else 1.0
        offset = -col_min * scale
        train_df[col] = train_df[col] * scale + offset
        test_df[col]  = test_df[col]  * scale + offset

    print(f"[8] Min-Max scaling applied to {len(NUMERIC_COLS)} numeric features:")
    for col, (mn, mx) in scale_params.items():
        print(f"    {col}: [{mn:.4f}, {mx:.4f}]")

    return train_df, test_df, scale_params


#  Step 8 — Column reorder and save
def reorder_and_save(train_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     train_out: str,
                     test_out:  str,
                     overwrite: bool = False):
    """
    Reorder columns to the canonical order and save as CSV.
    """
    # Desired column order (matches existing processed files)
    desired_order = [
        "person_age", "person_income", "person_emp_length",
        "loan_grade", "loan_amnt", "loan_int_rate",
        "loan_percent_income", "cb_person_default_on_file",
        "cb_person_cred_hist_length",
        "home_ownership_MORTGAGE", "home_ownership_OTHER",
        "home_ownership_OWN", "home_ownership_RENT",
        "loan_intent_DEBTCONSOLIDATION", "loan_intent_EDUCATION",
        "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL",
        "loan_intent_PERSONAL", "loan_intent_VENTURE",
        TARGET_COL,
    ]
    # Keep only columns that exist (handles potential OHE name variations)
    available     = set(train_df.columns)
    ordered_cols  = [c for c in desired_order if c in available]
    remaining     = [c for c in train_df.columns if c not in ordered_cols]
    final_cols    = ordered_cols + remaining

    train_df = train_df[final_cols]
    test_df  = test_df[final_cols]

    if os.path.exists(train_out) and not overwrite:
        print(f"\n[9] '{os.path.basename(train_out)}' already exists — skipping save.")
        print(f"    Run with overwrite=True to regenerate.")
    else:
        os.makedirs(os.path.dirname(train_out), exist_ok=True)
        train_df.to_csv(train_out, index=False, lineterminator="\r\n")
        test_df.to_csv(test_out, index=False, lineterminator="\r\n")
        print(f"\n[9] Saved:")
        print(f"    {train_out}  ({len(train_df)} rows × {len(final_cols)} columns)")
        print(f"    {test_out}   ({len(test_df)} rows × {len(final_cols)} columns)")

    return train_df, test_df


#  Main pipeline
def run_preprocessing(overwrite: bool = True):
    print("Credit Risk Dataset — Preprocessing Pipeline")

    # Step 1
    df = load_raw(RAW_PATH)

    # Step 2 — Stratified split on raw data to match the original reports
    train_df, test_df = stratified_split(df)

    # Step 3 — Remove outliers after the split
    train_df = remove_outliers(train_df)
    test_df = remove_outliers(test_df)

    # Step 4 — Label encode each split
    train_df = label_encode(train_df)
    test_df = label_encode(test_df)

    # Step 5 — One-hot encode each split, then align feature columns.
    train_df = one_hot_encode(train_df)
    test_df = one_hot_encode(test_df)
    train_df, test_df = align_feature_columns(train_df, test_df)

    # Step 6 — Impute missing values (train stats only)
    train_df, test_df = impute_missing(train_df, test_df)

    # Step 7 — Clip income outliers (train stats only)
    train_df, test_df, income_upper = clip_person_income(train_df, test_df)

    # Step 8 — Min-Max scale (train stats only)
    train_df, test_df, scale_params = minmax_scale(train_df, test_df)
    scale_params["person_income_iqr_upper"] = income_upper

    # Step 9 — Save
    train_df, test_df = reorder_and_save(
        train_df, test_df, TRAIN_OUT, TEST_OUT, overwrite=overwrite
    )

    print("\nPreprocessing complete.")
    print(f"  Final feature count : {train_df.shape[1] - 1}")
    print(f"  Train samples       : {len(train_df)}")
    print(f"  Test samples        : {len(test_df)}")
    print(f"  Train default rate  : {train_df[TARGET_COL].mean():.4f}")
    print(f"  Test  default rate  : {test_df[TARGET_COL].mean():.4f}")

    return train_df, test_df, scale_params


if __name__ == "__main__":
    run_preprocessing(overwrite=True)
