"""
=============================================================================
 mlp_ablation.py

 Part 1 - DELIBERATE OVERFITTING:
   Build a large MLP (much bigger than the 242 training patients need) with
   NO regularisation and show train accuracy running far ahead of
   validation accuracy.

 Part 2 - ABLATION TABLE:
   Using that SAME large architecture, compare four settings:
     1. none            (no dropout, no batch norm)
     2. dropout only
     3. batch norm only
     4. dropout + batch norm
   using 5-fold stratified CV on the training set, and report the
   train-validation accuracy gap for each - this is the table the rubric
   asks for.

 Reuses build_model() from mlp_framework.py so the only thing that
 changes between settings is the dropout/batchnorm arguments.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from logger import log_result
from mlp_framework import build_model

RANDOM_SEED = 42
PROCESSED_DIR = "data/processed"
FIG_DIR = "report/figures"

# Deliberately oversized for 242 patients / 20 features - plenty of
# capacity to memorise the training set if nothing stops it, while still
# training in a reasonable time on a CPU (no GPU needed for this project).
LARGE_HIDDEN = [128, 64]
EPOCHS = 120
BATCH_SIZE = 32
DROPOUT_RATE = 0.4
N_FOLDS = 5


def load_train():
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values.astype("float32")
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel().astype("float32")
    return X, y


# -----------------------------------------------------------------------
# Part 1: deliberate overfitting demo (single 80/20 split, so we can plot
# a clean train-vs-validation curve over epochs)
# -----------------------------------------------------------------------
def overfit_demo(X, y):
    print("=== Part 1: deliberately overfitting a large MLP ===")
    print(f"Architecture: [{X.shape[1]}] + {LARGE_HIDDEN} + [1], no regularisation\n")

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr_idx, va_idx = idx[:cut], idx[cut:]

    model = build_model(X.shape[1], hidden_sizes=LARGE_HIDDEN, dropout=0.0, batchnorm=False)
    history = model.fit(X[tr_idx], y[tr_idx], validation_data=(X[va_idx], y[va_idx]),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final train accuracy      : {final_train_acc:.3f}")
    print(f"Final validation accuracy : {final_val_acc:.3f}")
    print(f"Gap                       : {final_train_acc - final_val_acc:.3f}")

    log_result("mlp_ablation.py", "overfit_demo", f"hidden={LARGE_HIDDEN};epochs={EPOCHS}",
               RANDOM_SEED, "final_train_acc", final_train_acc)
    log_result("mlp_ablation.py", "overfit_demo", f"hidden={LARGE_HIDDEN};epochs={EPOCHS}",
               RANDOM_SEED, "final_val_acc", final_val_acc)
    log_result("mlp_ablation.py", "overfit_demo", f"hidden={LARGE_HIDDEN};epochs={EPOCHS}",
               RANDOM_SEED, "final_train_val_gap", final_train_acc - final_val_acc)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Deliberate overfitting: large MLP, no regularisation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "overfit_demo.png"), dpi=150)
    print(f"Saved plot -> {FIG_DIR}/overfit_demo.png\n")


# -----------------------------------------------------------------------
# Part 2: the 4-way ablation, each setting run with 5-fold stratified CV
# -----------------------------------------------------------------------
SETTINGS = {
    "none":            dict(dropout=0.0,          batchnorm=False),
    "dropout_only":    dict(dropout=DROPOUT_RATE,  batchnorm=False),
    "batchnorm_only":  dict(dropout=0.0,          batchnorm=True),
    "both":            dict(dropout=DROPOUT_RATE,  batchnorm=True),
}


def run_ablation(X, y):
    print("=== Part 2: dropout / batch norm ablation (5-fold stratified CV) ===")
    print(f"Architecture: [{X.shape[1]}] + {LARGE_HIDDEN} + [1] | "
          f"dropout rate when used = {DROPOUT_RATE}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows = []

    for setting_name, kwargs in SETTINGS.items():
        tr_accs, va_accs, va_aucs = [], [], []
        for tr_idx, va_idx in skf.split(X, y):
            model = build_model(X.shape[1], hidden_sizes=LARGE_HIDDEN, **kwargs)
            model.fit(X[tr_idx], y[tr_idx], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

            tr_acc = model.evaluate(X[tr_idx], y[tr_idx], verbose=0)[1]
            va_acc = model.evaluate(X[va_idx], y[va_idx], verbose=0)[1]
            va_prob = model.predict(X[va_idx], verbose=0).ravel()
            va_auc = roc_auc_score(y[va_idx], va_prob)

            tr_accs.append(tr_acc); va_accs.append(va_acc); va_aucs.append(va_auc)

        row = {
            "setting": setting_name,
            "train_acc": np.mean(tr_accs),
            "val_acc": np.mean(va_accs),
            "val_acc_std": np.std(va_accs),
            "val_auc": np.mean(va_aucs),
            "gap": np.mean(tr_accs) - np.mean(va_accs),
        }
        rows.append(row)
        print(f"{setting_name:<15} train {row['train_acc']:.3f} | "
              f"val {row['val_acc']:.3f} (+/-{row['val_acc_std']:.3f}) | "
              f"val AUC {row['val_auc']:.3f} | gap {row['gap']:.3f}")

        settings_str = f"hidden={LARGE_HIDDEN};dropout={kwargs['dropout']};batchnorm={kwargs['batchnorm']}"
        model_name = f"mlp_ablation_{setting_name}"
        log_result("mlp_ablation.py", model_name, settings_str, RANDOM_SEED, "cv_train_acc", row["train_acc"])
        log_result("mlp_ablation.py", model_name, settings_str, RANDOM_SEED, "cv_val_acc", row["val_acc"])
        log_result("mlp_ablation.py", model_name, settings_str, RANDOM_SEED, "cv_val_acc_std", row["val_acc_std"])
        log_result("mlp_ablation.py", model_name, settings_str, RANDOM_SEED, "cv_val_auc", row["val_auc"])
        log_result("mlp_ablation.py", model_name, settings_str, RANDOM_SEED, "cv_train_val_gap", row["gap"])

    # Save the ablation table as its own CSV too, for easy pasting into the report
    table = pd.DataFrame(rows)
    table_path = os.path.join(FIG_DIR, "..", "ablation_table.csv")
    table.to_csv(table_path, index=False)
    print(f"\nSaved ablation table -> report/ablation_table.csv")
    return table


def main():
    X, y = load_train()
    print(f"Training data: {X.shape[0]} patients, {X.shape[1]} features\n")

    overfit_demo(X, y)
    table = run_ablation(X, y)

    print("\n=== Summary table ===")
    print(table.round(3).to_string(index=False))
    print("\nSaved metrics -> experiments/results.csv")


if __name__ == "__main__":
    main()