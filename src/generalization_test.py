"""
=============================================================================
 generalization_test.py

 Trains on Cleveland (already done), tests on the Hungarian dataset - a
 different hospital, never seen during training or threshold tuning.

 This is NOT the project's official test set. It's a separate check of
 how well the Cleveland-trained model generalises to a different patient
 population and data-collection setup.

 What this script does:
   1. Downloads processed.hungarian.data if not already present.
   2. Parses it (same 14 columns as Cleveland, "?" marks missing values).
   3. Fills missing values using statistics learned from Cleveland
      (median for numeric columns, mode for categorical columns) - NOT
      from Hungarian itself, since we are simulating "a new hospital's
      data arriving", where we would not get to peek at its own stats.
   4. One-hot encodes and scales using the SAME encoder/scaler fitted on
      Cleveland in data_prep.py, so a category Hungarian doesn't have
      does not break anything, and a category Cleveland didn't have
      is dropped (the model was never trained to use it).
   5. Runs the final model (from evaluate.py) on this data and reports
      the same metrics as the Cleveland test set, for direct comparison.
=============================================================================
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

HUNGARIAN_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                  "heart-disease/processed.hungarian.data")
HUNGARIAN_PATH = os.path.join(RAW_DIR, "processed.hungarian.data")

# Same order as Cleveland's processed files and as data_prep.py's raw save
COLUMN_NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target_raw"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def download_hungarian():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(HUNGARIAN_PATH):
        print(f"Using existing file: {HUNGARIAN_PATH}")
        return
    print(f"Downloading {HUNGARIAN_URL} ...")
    try:
        urllib.request.urlretrieve(HUNGARIAN_URL, HUNGARIAN_PATH)
        print(f"Saved to {HUNGARIAN_PATH}")
    except Exception as e:
        raise SystemExit(
            f"\nCould not download the file automatically ({e}).\n"
            f"Please download it yourself from:\n  {HUNGARIAN_URL}\n"
            f"and save it as:\n  {HUNGARIAN_PATH}\nthen run this script again."
        )


def load_hungarian():
    df = pd.read_csv(HUNGARIAN_PATH, header=None, names=COLUMN_NAMES, na_values="?")
    print(f"Loaded {len(df)} Hungarian patients")
    print("Missing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df


def get_cleveland_fill_values():
    """
    Learn fill values (median for numeric, mode for categorical) from the
    FULL Cleveland dataset - this plays the role of "what a hospital
    deploying this model would already know from its training data",
    used to fill in whatever Hungarian is missing.
    """
    cleveland_path = os.path.join(RAW_DIR, "cleveland_raw.csv")
    if not os.path.exists(cleveland_path):
        raise SystemExit(
            f"Could not find {cleveland_path}. Run data_prep.py first "
            f"(it saves this file as a side effect)."
        )
    df = pd.read_csv(cleveland_path)
    fill_values = {}
    for col in NUMERIC_COLS:
        fill_values[col] = df[col].median()
    for col in CATEGORICAL_COLS:
        fill_values[col] = df[col].mode()[0]
    return fill_values


def preprocess(df, fill_values):
    df = df.copy()
    df["target"] = (df["target_raw"] > 0).astype(int)
    df = df.drop(columns=["target_raw"])

    # Fill ALL missing values using Cleveland-derived statistics (not
    # Hungarian's own, since a real deployment wouldn't get to "peek").
    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    y = df["target"].values
    X_raw = df.drop(columns=["target"])

    # One-hot encode, then force the columns to match what the model was
    # trained on: any Cleveland dummy column Hungarian's data doesn't
    # produce gets added as all-zero; any category Hungarian has that
    # Cleveland never had gets dropped (the model has no weight for it).
    X_encoded = pd.get_dummies(X_raw, columns=CATEGORICAL_COLS, drop_first=True)
    feature_columns = joblib.load(os.path.join(PROCESSED_DIR, "feature_columns.joblib"))
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)
    X_encoded = X_encoded.astype(float)

    # Scale numeric columns with the SAME scaler fitted on Cleveland train
    scaler = joblib.load(os.path.join(PROCESSED_DIR, "scaler.joblib"))
    X_encoded[NUMERIC_COLS] = scaler.transform(X_encoded[NUMERIC_COLS])

    return X_encoded.values.astype("float32"), y.astype("float32")


def main():
    download_hungarian()
    df = load_hungarian()
    fill_values = get_cleveland_fill_values()
    X, y = preprocess(df, fill_values)
    print(f"\nProcessed Hungarian data: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Disease prevalence in Hungarian data: {y.mean():.3f} "
          f"(Cleveland train was ~0.46)")

    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "final_model.keras"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "threshold.joblib"))
    print(f"Using threshold from evaluate.py: {threshold:.3f}")

    prob = model.predict(X, verbose=0).ravel()
    pred = (prob >= threshold).astype(int)

    auc = roc_auc_score(y, prob)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print("\n=== Hungarian generalization results ===")
    print(f"ROC-AUC     : {auc:.3f}")
    print(f"Accuracy    : {accuracy:.3f}")
    print(f"Recall      : {recall:.3f}")
    print(f"Specificity : {specificity:.3f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
 


if __name__ == "__main__":
    main()