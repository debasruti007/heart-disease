"""
=============================================================================
 mlp_framework.py
 The SAME architecture as mlp_scratch.py, this time built with Keras.

 Two things happen here:
   1. MATCH CHECK: load the exact weights saved by mlp_scratch.py into a
      Keras model and confirm both give (almost) identical predictions.
      This is the "verified against framework output" requirement.
   2. Its own 5-fold stratified cross-validation, trained fresh with Keras'
      own optimiser, so we have a framework baseline to compare against
      the from-scratch numbers, and to extend with Dropout/BatchNorm later.

 No Dropout or BatchNorm yet - that ablation is the next script
 (mlp_ablation.py), which reuses build_model() from this file.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from logger import log_result

RANDOM_SEED = 42
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

HIDDEN_SIZES = [16, 8]      # must match mlp_scratch.py
LEARNING_RATE = 0.05
EPOCHS = 200
BATCH_SIZE = 32
N_FOLDS = 5


def load_train():
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values.astype("float32")
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel().astype("float32")
    return X, y


def build_model(n_features, hidden_sizes=HIDDEN_SIZES, dropout=0.0, batchnorm=False,
                 seed=RANDOM_SEED):
    """
    Same shape as the scratch MLP: Dense+ReLU hidden layers, Dense+Sigmoid output.
    dropout=0.0 and batchnorm=False reproduces the plain (unregularised) scratch
    model exactly. mlp_ablation.py calls this same function with other settings.
    """
    tf.keras.utils.set_random_seed(seed)
    model = models.Sequential(name="heart_mlp")
    model.add(layers.Input(shape=(n_features,)))
    for h in hidden_sizes:
        model.add(layers.Dense(h, use_bias=not batchnorm,
                                kernel_initializer="he_normal"))
        if batchnorm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------------------------------------------------
# 1. Match check: load mlp_scratch's saved weights into an identical
#    Keras model (no dropout/batchnorm, since the scratch model has none)
#    and compare predictions on the same data.
# -----------------------------------------------------------------------
def match_check():
    print("=== Match check: Keras vs from-scratch, same weights ===")
    weights_path = os.path.join(MODEL_DIR, "mlp_scratch_weights.npz")
    if not os.path.exists(weights_path):
        print(f"Could not find {weights_path}. Run mlp_scratch.py first.")
        return False

    d = np.load(weights_path)
    sizes = list(d["layer_sizes"])          # e.g. [20, 16, 8, 1]
    n_features = int(sizes[0])
    hidden = [int(s) for s in sizes[1:-1]]

    keras_model = build_model(n_features, hidden_sizes=hidden, dropout=0.0, batchnorm=False)

    # Keras Dense layers are at positions 0, 2, 4, ... because an
    # Activation layer sits between each pair (see build_model above).
    dense_layers = [l for l in keras_model.layers if isinstance(l, layers.Dense)]
    n_layers = len(dense_layers)
    for l in range(n_layers):
        dense_layers[l].set_weights([d[f"W{l}"], d[f"b{l}"]])

    X, y = load_train()
    from mlp_scratch import MLP
    scratch_model = MLP(sizes)
    for l in range(n_layers):
        scratch_model.W[l] = d[f"W{l}"]
        scratch_model.b[l] = d[f"b{l}"]

    scratch_pred = scratch_model.predict_proba(X)
    keras_pred = keras_model.predict(X, verbose=0).ravel()

    max_diff = float(np.max(np.abs(scratch_pred - keras_pred)))
    mean_diff = float(np.mean(np.abs(scratch_pred - keras_pred)))
    print(f"Max  |difference| across {len(X)} patients : {max_diff:.2e}")
    print(f"Mean |difference| across {len(X)} patients : {mean_diff:.2e}")
    status = "PASS" if max_diff < 1e-4 else "FAIL"
    print(f"Match check: {status}")

    log_result("mlp_framework.py", "match_check", f"n_layers={n_layers}",
               RANDOM_SEED, "max_abs_diff", max_diff)
    log_result("mlp_framework.py", "match_check", f"n_layers={n_layers}",
               RANDOM_SEED, "mean_abs_diff", mean_diff)
    return status == "PASS"


# -----------------------------------------------------------------------
# 2. Keras' own 5-fold stratified cross-validation (fresh training,
#    no dropout/batchnorm - this is the plain framework baseline)
# -----------------------------------------------------------------------
def cross_validate():
    X, y = load_train()
    n_features = X.shape[1]
    print(f"\n=== {N_FOLDS}-fold stratified CV (Keras, no regularisation) ===")
    print(f"Architecture: [{n_features}] + {HIDDEN_SIZES} + [1] | "
          f"lr={LEARNING_RATE} | epochs={EPOCHS} | batch={BATCH_SIZE}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    tr_accs, va_accs, va_aucs = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = build_model(n_features, dropout=0.0, batchnorm=False)
        model.fit(X[tr_idx], y[tr_idx], epochs=EPOCHS, batch_size=BATCH_SIZE,
                  verbose=0)

        tr_acc = model.evaluate(X[tr_idx], y[tr_idx], verbose=0)[1]
        va_loss, va_acc = model.evaluate(X[va_idx], y[va_idx], verbose=0)
        va_prob = model.predict(X[va_idx], verbose=0).ravel()
        va_auc = roc_auc_score(y[va_idx], va_prob)

        tr_accs.append(tr_acc); va_accs.append(va_acc); va_aucs.append(va_auc)
        print(f"Fold {fold}: train acc {tr_acc:.3f} | val acc {va_acc:.3f} | val AUC {va_auc:.3f}")

    print(f"\nMean train acc : {np.mean(tr_accs):.3f}")
    print(f"Mean val acc   : {np.mean(va_accs):.3f}  (std {np.std(va_accs):.3f})")
    print(f"Mean val AUC   : {np.mean(va_aucs):.3f}  (std {np.std(va_aucs):.3f})")
    print(f"Train-val accuracy gap: {np.mean(tr_accs) - np.mean(va_accs):.3f}")

    settings = (f"layers=[{n_features}]+{HIDDEN_SIZES}+[1];lr={LEARNING_RATE};"
                f"epochs={EPOCHS};batch={BATCH_SIZE};folds={N_FOLDS};"
                f"dropout=0;batchnorm=False")
    model_name = "mlp_keras_plain"
    log_result("mlp_framework.py", model_name, settings, RANDOM_SEED, "cv_train_acc", np.mean(tr_accs))
    log_result("mlp_framework.py", model_name, settings, RANDOM_SEED, "cv_val_acc", np.mean(va_accs))
    log_result("mlp_framework.py", model_name, settings, RANDOM_SEED, "cv_val_acc_std", np.std(va_accs))
    log_result("mlp_framework.py", model_name, settings, RANDOM_SEED, "cv_val_auc", np.mean(va_aucs))
    log_result("mlp_framework.py", model_name, settings, RANDOM_SEED, "cv_train_val_gap",
               np.mean(tr_accs) - np.mean(va_accs))


def main():
    ok = match_check()
    if not ok:
        print("\nStopping: the Keras and from-scratch models do not agree. "
              "Do not continue until match_check() passes.")
        return
    cross_validate()
    print("\nSaved metrics -> experiments/results.csv")


if __name__ == "__main__":
    main()