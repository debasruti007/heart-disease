"""
=============================================================================
 evaluate.py

 Takes the winning ablation setting (dropout + batch norm, from
 mlp_ablation.py's results) as the FINAL model, and:

   1. Gets out-of-fold (OOF) predictions via 5-fold stratified CV on the
      TRAINING set only. "Out-of-fold" means: for every patient, the
      probability used is from a model that never saw that patient during
      training. This gives an honest, full-training-set-sized set of
      predictions without ever touching the test set.
   2. Uses those OOF predictions to:
        - report the mean cross-validated ROC-AUC (success criterion)
        - tune a decision threshold that reaches recall >= 0.85 while
          keeping specificity as high as possible
        - draw a calibration (reliability) plot
   3. Trains ONE final model on the FULL training set with the same
      settings, and evaluates it on the TEST set EXACTLY ONCE, using the
      threshold chosen in step 1. This is the only test-set evaluation
      in the whole project, as the brief requires.
   4. Saves the final model and the chosen threshold to disk, for the
      web app to load later.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import joblib

from logger import log_result
from mlp_framework import build_model

RANDOM_SEED = 42
PROCESSED_DIR = "data/processed"
FIG_DIR = "report/figures"
MODEL_DIR = "models"

# The winning setting from mlp_ablation.py's table: dropout + batch norm.
# If your own ablation table names a different winner, change these two
# lines to match - everything else in this script stays the same.
FINAL_HIDDEN = [128, 64]
FINAL_DROPOUT = 0.4
FINAL_BATCHNORM = True

EPOCHS = 120
BATCH_SIZE = 32
N_FOLDS = 5
TARGET_RECALL = 0.85


def load_train():
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values.astype("float32")
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel().astype("float32")
    return X, y


def load_test():
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv")).values.astype("float32")
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel().astype("float32")
    return X, y


# -----------------------------------------------------------------------
# 1. Out-of-fold predictions via 5-fold stratified CV (training set only)
# -----------------------------------------------------------------------
def get_oof_predictions(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = np.zeros(len(X), dtype=float)
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = build_model(X.shape[1], hidden_sizes=FINAL_HIDDEN,
                             dropout=FINAL_DROPOUT, batchnorm=FINAL_BATCHNORM)
        model.fit(X[tr_idx], y[tr_idx], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        prob = model.predict(X[va_idx], verbose=0).ravel()
        oof_prob[va_idx] = prob
        fold_auc = roc_auc_score(y[va_idx], prob)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold}: val ROC-AUC = {fold_auc:.3f}")

    return oof_prob, fold_aucs


# -----------------------------------------------------------------------
# 2. Threshold tuning: highest specificity subject to recall >= target
# -----------------------------------------------------------------------
def tune_threshold(y_true, y_prob, target_recall=TARGET_RECALL):
    """
    roc_curve gives, for a sweep of thresholds:
      fpr (false positive rate)  = 1 - specificity
      tpr (true positive rate)   = recall
    We keep only thresholds where recall >= target, then among those pick
    the one with the lowest fpr (= highest specificity). This gives the
    threshold that still catches enough disease cases while raising as
    few unnecessary alarms as possible.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = tpr >= target_recall
    if not ok.any():
        # No threshold reaches the target recall (can happen on small/noisy
        # data) - fall back to the threshold with the highest recall available.
        best_idx = int(np.argmax(tpr))
        print(f"  WARNING: no threshold reached recall >= {target_recall}. "
              f"Using the best available recall instead.")
    else:
        candidate_idxs = np.where(ok)[0]
        best_idx = candidate_idxs[np.argmin(fpr[candidate_idxs])]

    chosen_threshold = float(thresholds[best_idx])
    # roc_curve's first threshold can be > 1 (a sentinel value); clip to a
    # sane probability range just in case that edge case is selected.
    chosen_threshold = float(np.clip(chosen_threshold, 0.0, 1.0))
    return chosen_threshold, fpr[best_idx], tpr[best_idx]


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")       # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"accuracy": accuracy, "recall": recall, "specificity": specificity,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# -----------------------------------------------------------------------
# 3. Calibration plot (reliability diagram) from OOF predictions
# -----------------------------------------------------------------------
def plot_calibration(y_true, y_prob):
    os.makedirs(FIG_DIR, exist_ok=True)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", label="model (OOF predictions)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of patients with disease")
    plt.title("Calibration plot (5-fold out-of-fold predictions)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "calibration_plot.png"), dpi=150)
    print(f"Saved plot -> {FIG_DIR}/calibration_plot.png")


def plot_roc(y_true, y_prob, auc, tag, fname):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (recall)")
    plt.title(f"ROC curve - {tag}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    print(f"Saved plot -> {FIG_DIR}/{fname}")


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------
def main():
    X_train, y_train = load_train()
    X_test, y_test = load_test()
    print(f"Final model settings: hidden={FINAL_HIDDEN}, dropout={FINAL_DROPOUT}, "
          f"batchnorm={FINAL_BATCHNORM}\n")

    # ---- Step 1: OOF predictions on the training set ----
    print("=== 5-fold out-of-fold predictions (training set only) ===")
    oof_prob, fold_aucs = get_oof_predictions(X_train, y_train)
    mean_cv_auc = float(np.mean(fold_aucs))
    print(f"\nMean cross-validated ROC-AUC: {mean_cv_auc:.3f} "
          f"(target: >= 0.88)")

    # ---- Step 2: tune threshold on OOF predictions ----
    threshold, fpr_at_thresh, recall_at_thresh = tune_threshold(y_train, oof_prob)
    cv_metrics = metrics_at_threshold(y_train, oof_prob, threshold)
    print(f"\nChosen threshold: {threshold:.3f}")
    print(f"At this threshold on CV (training) data:")
    print(f"  Accuracy    : {cv_metrics['accuracy']:.3f}")
    print(f"  Recall      : {cv_metrics['recall']:.3f}  (target: >= 0.85)")
    print(f"  Specificity : {cv_metrics['specificity']:.3f}")
    print(f"  Confusion matrix: TN={cv_metrics['tn']} FP={cv_metrics['fp']} "
          f"FN={cv_metrics['fn']} TP={cv_metrics['tp']}")

    # ---- Step 3: calibration + ROC plots from OOF predictions ----
    plot_calibration(y_train, oof_prob)
    plot_roc(y_train, oof_prob, mean_cv_auc, "5-fold OOF (training set)", "roc_curve_cv.png")

    log_result("evaluate.py", "final_model", "oof", RANDOM_SEED, "mean_cv_roc_auc", mean_cv_auc)
    log_result("evaluate.py", "final_model", "oof", RANDOM_SEED, "chosen_threshold", threshold)
    log_result("evaluate.py", "final_model", "oof", RANDOM_SEED, "cv_recall_at_threshold", cv_metrics["recall"])
    log_result("evaluate.py", "final_model", "oof", RANDOM_SEED, "cv_specificity_at_threshold", cv_metrics["specificity"])
    log_result("evaluate.py", "final_model", "oof", RANDOM_SEED, "cv_accuracy_at_threshold", cv_metrics["accuracy"])

    # ---- Step 4: train ONE final model on the FULL training set ----
    print("\n=== Training final model on the full training set ===")
    final_model = build_model(X_train.shape[1], hidden_sizes=FINAL_HIDDEN,
                               dropout=FINAL_DROPOUT, batchnorm=FINAL_BATCHNORM)
    final_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model.save(os.path.join(MODEL_DIR, "final_model.keras"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "threshold.joblib"))
    print(f"Saved model -> {MODEL_DIR}/final_model.keras")
    print(f"Saved threshold -> {MODEL_DIR}/threshold.joblib")

    # ---- Step 5: the ONE AND ONLY test-set evaluation ----
    print("\n=== FINAL test-set evaluation (test set touched once, here) ===")
    test_prob = final_model.predict(X_test, verbose=0).ravel()
    test_auc = roc_auc_score(y_test, test_prob)
    test_metrics = metrics_at_threshold(y_test, test_prob, threshold)

    print(f"Test ROC-AUC   : {test_auc:.3f}")
    print(f"Test Accuracy  : {test_metrics['accuracy']:.3f}")
    print(f"Test Recall    : {test_metrics['recall']:.3f}")
    print(f"Test Specificity: {test_metrics['specificity']:.3f}")
    print(f"Confusion matrix: TN={test_metrics['tn']} FP={test_metrics['fp']} "
          f"FN={test_metrics['fn']} TP={test_metrics['tp']}")

    plot_roc(y_test, test_prob, test_auc, "final test set", "roc_curve_test.png")

    log_result("evaluate.py", "final_model", "test_set", RANDOM_SEED, "test_roc_auc", test_auc)
    log_result("evaluate.py", "final_model", "test_set", RANDOM_SEED, "test_accuracy", test_metrics["accuracy"])
    log_result("evaluate.py", "final_model", "test_set", RANDOM_SEED, "test_recall", test_metrics["recall"])
    log_result("evaluate.py", "final_model", "test_set", RANDOM_SEED, "test_specificity", test_metrics["specificity"])

    print("\nSaved metrics -> experiments/results.csv")
    print("\nREMINDER: the test set has now been evaluated. Do not run this "
          "script again with a changed model and re-check test performance - "
          "that would break the 'test set used only once' rule.")


if __name__ == "__main__":
    main()