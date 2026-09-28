"""
=============================================================================
 data_prep.py
 Loads the UCI Cleveland Heart Disease dataset, cleans it, encodes it,
 and splits it into TRAIN (80%) and TEST (20%).

 IMPORTANT (no data leakage):
   - Missing-value fill values and the scaler are learned from TRAIN only,
     then applied to TEST.
   - The test set is never used until the final evaluation.
   - Validation is done later using 5-fold stratified cross-validation
     inside the training set.
=============================================================================
"""

import os
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Same seed used in EVERY script of this project
RANDOM_SEED = 42

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Categorical columns (stored as numbers, but they are categories)
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
# Truly continuous columns
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Columns that contain missing values in the Cleveland data
MISSING_COLS = ["ca", "thal"]


def load_raw_data():
    """Download Cleveland data (id=45) and return one DataFrame."""
    heart = fetch_ucirepo(id=45)
    X = heart.data.features.copy()
    y = heart.data.targets.copy()

    df = X.copy()
    df["target_raw"] = y.values.ravel()

    # keep an untouched copy of the raw data
    df.to_csv(os.path.join(RAW_DIR, "cleveland_raw.csv"), index=False)
    return df


def make_binary_target(df):
    """Convert target 0-4 into 0 = no disease, 1 = disease present."""
    df = df.copy()
    df["target"] = (df["target_raw"] > 0).astype(int)
    df = df.drop(columns=["target_raw"])
    return df


def main():
    print("Loading raw data...")
    df = make_binary_target(load_raw_data())
    print(f"Total patients: {len(df)}")
    print("Missing values per column (before cleaning):")
    print(df[MISSING_COLS].isnull().sum())

    # ---------------------------------------------------------------
    # 1) Split FIRST, before learning anything from the data
    # ---------------------------------------------------------------
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_SEED
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # ---------------------------------------------------------------
    # 2) Missing values: fill with the most frequent value found in TRAIN
    # ---------------------------------------------------------------
    impute_values = {col: train_df[col].mode()[0] for col in MISSING_COLS}
    for col, value in impute_values.items():
        train_df[col] = train_df[col].fillna(value)
        test_df[col] = test_df[col].fillna(value)

    y_train = train_df["target"]
    y_test = test_df["target"]
    X_train = train_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

    # ---------------------------------------------------------------
    # 3) One-hot encode categorical columns.
    #    Done on train+test together only so both get identical columns
    #    (this uses no target information, so there is no leakage).
    # ---------------------------------------------------------------
    n_train = len(X_train)
    combined = pd.concat([X_train, X_test], ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS, drop_first=True)
    combined = combined.astype(float)

    X_train = combined.iloc[:n_train].copy().reset_index(drop=True)
    X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

    # ---------------------------------------------------------------
    # 4) Scale numeric columns: fit on TRAIN only, apply to TEST
    # ---------------------------------------------------------------
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    # ---------------------------------------------------------------
    # 5) Save everything
    # ---------------------------------------------------------------
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    # Needed later by the web app and the Hungarian generalization test
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.joblib"))
    joblib.dump(NUMERIC_COLS, os.path.join(PROCESSED_DIR, "numeric_cols.joblib"))
    joblib.dump(list(X_train.columns), os.path.join(PROCESSED_DIR, "feature_columns.joblib"))
    joblib.dump(impute_values, os.path.join(PROCESSED_DIR, "impute_values.joblib"))

    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Number of features after encoding: {X_train.shape[1]}")
    print("Train class balance:")
    print(y_train.value_counts(normalize=True).round(3))
    print("Test class balance:")
    print(y_test.value_counts(normalize=True).round(3))
    print("\nDone. Files saved in data/processed/")


if __name__ == "__main__":
    main()