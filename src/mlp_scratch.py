"""
=============================================================================
 mlp_scratch.py
 Multi-Layer Perceptron with BACKPROPAGATION written FROM SCRATCH (NumPy only).

 Network:   input -> Dense+ReLU -> Dense+ReLU -> ... -> Dense+Sigmoid (1 output)
 Loss:      binary cross-entropy
 Optimiser: mini-batch gradient descent (plain SGD)

 What this script does
   1. Gradient check  : proves our backprop maths matches numerical gradients
   2. 5-fold stratified cross-validation on the TRAINING set only
   3. Plots train vs validation loss (so we can see overfitting)
   4. Trains one final model on the full training set and saves its weights,
      so the Keras version (next step) can be checked against it

 The test set is NOT touched here.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save plots to files, no window needed
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from logger import log_result

RANDOM_SEED = 42
PROCESSED_DIR = "data/processed"
FIG_DIR = "report/figures"
MODEL_DIR = "models"

HIDDEN_SIZES = [16, 8]      # two hidden layers
LEARNING_RATE = 0.05
EPOCHS = 200
BATCH_SIZE = 32
N_FOLDS = 5


# -----------------------------------------------------------------------
# 1. Activation functions and their derivatives
# -----------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)          # avoids overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


# -----------------------------------------------------------------------
# 2. The MLP class
# -----------------------------------------------------------------------
class MLP:
    def __init__(self, layer_sizes, seed=42):
        """
        layer_sizes example: [20, 16, 8, 1]
          = 20 inputs, hidden layers of 16 and 8 neurons, 1 output neuron.

        Weight W[l] has shape (n_in, n_out) and bias b[l] has shape (n_out,),
        the same layout as a Keras Dense layer, so weights can be copied
        between this model and Keras later.
        """
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialisation (suits ReLU): std = sqrt(2 / n_in)
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))
        self.cache_A = []      # activations of every layer (A[0] = input X)
        self.cache_Z = []      # pre-activations of every layer

    # ---------------- forward pass ----------------
    def forward(self, X):
        self.cache_A = [X]
        self.cache_Z = []
        A = X
        L = len(self.W)
        for l in range(L):
            Z = A @ self.W[l] + self.b[l]
            A = sigmoid(Z) if l == L - 1 else relu(Z)
            self.cache_Z.append(Z)
            self.cache_A.append(A)
        return A                               # shape (m, 1): probabilities

    # ---------------- loss ----------------
    @staticmethod
    def loss(y_prob, y):
        """Binary cross-entropy, averaged over the batch."""
        y = y.reshape(-1, 1)
        p = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    # ---------------- backward pass (backpropagation) ----------------
    def backward(self, y):
        """
        Must be called right after forward() on the same batch.
        Returns gradients dW, db (lists, same shapes as W and b).

        Key facts used:
          * For sigmoid output + binary cross-entropy, the gradient of the
            loss w.r.t. the output pre-activation simplifies to (A - y) / m.
          * For a hidden layer: dZ = dA * relu'(Z)
          * dW = A_prev^T @ dZ ,  db = sum of dZ over the batch
          * dA_prev = dZ @ W^T   (this is the "chain rule" step backwards)
        """
        m = y.shape[0]
        y = y.reshape(-1, 1)
        L = len(self.W)
        dW = [None] * L
        db = [None] * L

        dZ = (self.cache_A[-1] - y) / m            # output layer
        for l in range(L - 1, -1, -1):
            dW[l] = self.cache_A[l].T @ dZ
            db[l] = dZ.sum(axis=0)
            if l > 0:
                dA = dZ @ self.W[l].T              # pass gradient to previous layer
                dZ = dA * relu_grad(self.cache_Z[l - 1])
        return dW, db

    # ---------------- parameter update ----------------
    def step(self, dW, db, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    # ---------------- predictions ----------------
    def predict_proba(self, X):
        return self.forward(X).ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---------------- training loop ----------------
    def fit(self, X, y, lr=0.05, epochs=200, batch_size=32, seed=42,
            X_val=None, y_val=None):
        """
        Mini-batch gradient descent. Returns (train_loss_history, val_loss_history).
        Validation data is only used to record the loss curve, never to
        change the weights.
        """
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        train_hist, val_hist = [], []

        for epoch in range(epochs):
            order = rng.permutation(n)                 # new shuffle each epoch
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                self.forward(X[idx])
                dW, db = self.backward(y[idx])
                self.step(dW, db, lr)

            # record loss on the full train set (and validation set if given)
            train_hist.append(self.loss(self.forward(X), y))
            if X_val is not None:
                val_hist.append(self.loss(self.forward(X_val), y_val))
        return train_hist, val_hist

    # ---------------- save / load ----------------
    def save(self, path):
        arrays = {}
        for l in range(len(self.W)):
            arrays[f"W{l}"] = self.W[l]
            arrays[f"b{l}"] = self.b[l]
        np.savez(path, layer_sizes=np.array(self.layer_sizes), **arrays)


# -----------------------------------------------------------------------
# 3. Gradient check: compare backprop with numerical derivatives
# -----------------------------------------------------------------------
def gradient_check(seed=0, eps=1e-5):
    """
    Numerical gradient of every single weight:
        (loss(w + eps) - loss(w - eps)) / (2 * eps)
    If backprop is correct, this matches our analytic gradient almost exactly.
    Uses a small random network and a small random batch, in float64.
    Returns the relative error (should be around 1e-8 or smaller).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 6))
    y = rng.integers(0, 2, size=20).astype(float)
    net = MLP([6, 5, 4, 1], seed=seed)
    # Random non-zero biases: with zero biases a neuron input can land exactly
    # on 0, which is the sharp corner of ReLU where a numerical derivative is
    # not meaningful. (This affects only the check, not the real training.)
    for bias in net.b:
        bias[:] = rng.normal(0.0, 0.5, size=bias.shape)

    # analytic gradients from backpropagation
    net.forward(X)
    dW, db = net.backward(y)
    analytic = np.concatenate([g.ravel() for g in dW + db])

    # numerical gradients, one parameter at a time
    params = net.W + net.b            # list of arrays (edited in place)
    numeric = []
    for p in params:
        flat = p.reshape(-1)          # view on p, so edits change p
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            loss_plus = net.loss(net.forward(X), y)
            flat[i] = old - eps
            loss_minus = net.loss(net.forward(X), y)
            flat[i] = old
            numeric.append((loss_plus - loss_minus) / (2 * eps))
    numeric = np.array(numeric)

    rel_error = (np.linalg.norm(analytic - numeric)
                 / (np.linalg.norm(analytic) + np.linalg.norm(numeric)))
    return rel_error


# -----------------------------------------------------------------------
# 4. Data loading (training set only)
# -----------------------------------------------------------------------
def load_train():
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values.astype(float)
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel().astype(int)
    return X, y


# -----------------------------------------------------------------------
# 5. Run everything
# -----------------------------------------------------------------------
def main():
    # ---- Step 1: gradient check ----
    print("=== Gradient check ===")
    err = gradient_check()
    status = "PASS" if err < 1e-6 else "FAIL"
    print(f"Relative error between backprop and numerical gradient: {err:.2e}  -> {status}")
    if status == "FAIL":
        print("Backprop is wrong or a ReLU kink was hit. Do NOT continue until this passes.")
        return

    # ---- Step 2: 5-fold stratified cross-validation ----
    X, y = load_train()
    sizes = [X.shape[1]] + HIDDEN_SIZES + [1]
    print(f"\n=== {N_FOLDS}-fold stratified CV on training set ===")
    print(f"Architecture: {sizes} | lr={LEARNING_RATE} | epochs={EPOCHS} | batch={BATCH_SIZE}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    tr_accs, va_accs, va_aucs = [], [], []
    tr_curves, va_curves = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        net = MLP(sizes, seed=RANDOM_SEED)
        tr_hist, va_hist = net.fit(X[tr_idx], y[tr_idx], LEARNING_RATE, EPOCHS,
                                   BATCH_SIZE, RANDOM_SEED,
                                   X_val=X[va_idx], y_val=y[va_idx])
        tr_acc = np.mean(net.predict(X[tr_idx]) == y[tr_idx])
        va_acc = np.mean(net.predict(X[va_idx]) == y[va_idx])
        va_auc = roc_auc_score(y[va_idx], net.predict_proba(X[va_idx]))
        tr_accs.append(tr_acc); va_accs.append(va_acc); va_aucs.append(va_auc)
        tr_curves.append(tr_hist); va_curves.append(va_hist)
        print(f"Fold {fold}: train acc {tr_acc:.3f} | val acc {va_acc:.3f} | val AUC {va_auc:.3f}")

    print(f"\nMean train acc : {np.mean(tr_accs):.3f}")
    print(f"Mean val acc   : {np.mean(va_accs):.3f}  (std {np.std(va_accs):.3f})")
    print(f"Mean val AUC   : {np.mean(va_aucs):.3f}  (std {np.std(va_aucs):.3f})")
    print(f"Train-val accuracy gap: {np.mean(tr_accs) - np.mean(va_accs):.3f}")

    settings = (f"layers={sizes};lr={LEARNING_RATE};epochs={EPOCHS};"
                f"batch={BATCH_SIZE};folds={N_FOLDS}")
    model_name = "mlp_scratch"
    log_result("mlp_scratch.py", model_name, settings, RANDOM_SEED, "grad_check_rel_error", err)
    log_result("mlp_scratch.py", model_name, settings, RANDOM_SEED, "cv_train_acc", np.mean(tr_accs))
    log_result("mlp_scratch.py", model_name, settings, RANDOM_SEED, "cv_val_acc", np.mean(va_accs))
    log_result("mlp_scratch.py", model_name, settings, RANDOM_SEED, "cv_val_acc_std", np.std(va_accs))
    log_result("mlp_scratch.py", model_name, settings, RANDOM_SEED, "cv_val_auc", np.mean(va_aucs))
    log_result("mlp_scratch.py", model_name, settings, RANDOM_SEED, "cv_train_val_gap",
               np.mean(tr_accs) - np.mean(va_accs))

    # ---- Step 3: loss-curve plot (average over the 5 folds) ----
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(np.mean(tr_curves, axis=0), label="train loss")
    plt.plot(np.mean(va_curves, axis=0), label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("MLP from scratch: train vs validation loss (mean of 5 folds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mlp_scratch_loss_curves.png"), dpi=150)
    print(f"\nSaved plot -> {FIG_DIR}/mlp_scratch_loss_curves.png")

    # ---- Step 4: final model on the FULL training set, save weights ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    final = MLP(sizes, seed=RANDOM_SEED)
    final.fit(X, y, LEARNING_RATE, EPOCHS, BATCH_SIZE, RANDOM_SEED)
    final.save(os.path.join(MODEL_DIR, "mlp_scratch_weights.npz"))
    print(f"Saved weights -> {MODEL_DIR}/mlp_scratch_weights.npz")
    print("Saved metrics -> experiments/results.csv")


if __name__ == "__main__":
    main()