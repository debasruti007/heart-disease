"""
=============================================================================
 perceptron.py
 Perceptron learning rule FROM SCRATCH (NumPy only) on the heart disease
 data, with four activation functions: step, sigmoid, tanh, linear.

 Same update rule as in class:
        w <- w + lr * error * x
        b <- b + lr * error
 where error = target - output.

 Evaluation uses 5-fold stratified cross-validation on the TRAINING set
 only. The test set is not touched here.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save plots to files, no window needed
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from logger import log_result

RANDOM_SEED = 42
PROCESSED_DIR = "data/processed"
FIG_DIR = "report/figures"
LEARNING_RATE = 0.01
MAX_EPOCHS = 200
N_FOLDS = 5


# -----------------------------------------------------------------------
# 1. Activation functions (vectorised versions of the ones from class)
# -----------------------------------------------------------------------
def step(z):
    return np.where(z >= 0, 1.0, 0.0)

def sigmoid(z):
    z = np.clip(z, -500, 500)              # avoids overflow in exp
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def linear(z):
    return z

ACTIVATIONS = {"step": step, "sigmoid": sigmoid, "tanh": tanh, "linear": linear}

# Output value at or above which we predict "disease present".
# sigmoid/step/linear work with targets 0/1 -> threshold 0.5
# tanh works with targets -1/+1            -> threshold 0
THRESHOLDS = {"step": 0.5, "sigmoid": 0.5, "tanh": 0.0, "linear": 0.5}


# -----------------------------------------------------------------------
# 2. The Perceptron class
# -----------------------------------------------------------------------
class Perceptron:
    def __init__(self, n_inputs, activation, learning_rate=0.01,
                 epochs=200, seed=42):
        self.name = activation
        self.activation = ACTIVATIONS[activation]
        self.threshold = THRESHOLDS[activation]
        self.lr = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = np.zeros(n_inputs)   # start at zero, like in class
        self.bias = 0.0
        self.errors_per_epoch = []          # misclassifications after each epoch
        self.epochs_run = 0
        self.converged = False

    def raw_output(self, X):
        return self.activation(X @ self.weights + self.bias)

    def predict(self, X):
        """Hard 0/1 class prediction."""
        return (self.raw_output(X) >= self.threshold).astype(int)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)

        # tanh outputs values in (-1, 1), so its targets must be -1 / +1
        if self.name == "tanh":
            targets = np.where(y == 1, 1.0, -1.0)
        else:
            targets = y.astype(float)

        self.errors_per_epoch = []
        self.converged = False

        for epoch in range(self.epochs):
            # visit the training examples in a new random order every epoch
            for i in rng.permutation(len(X)):
                xi = X[i]
                out = float(self.activation(np.dot(self.weights, xi) + self.bias))
                error = targets[i] - out
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

            n_wrong = int(np.sum(self.predict(X) != y))
            self.errors_per_epoch.append(n_wrong)
            self.epochs_run = epoch + 1

            if n_wrong == 0:               # perfect on the training data
                self.converged = True
                break
        return self


# -----------------------------------------------------------------------
# 3. Data loading (training set only)
# -----------------------------------------------------------------------
def load_train():
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values.astype(float)
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel().astype(int)
    return X, y


# -----------------------------------------------------------------------
# 4. 5-fold stratified cross-validation for one activation
# -----------------------------------------------------------------------
def cross_validate(X, y, activation):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    train_accs, val_accs, epochs_used, n_converged = [], [], [], 0

    for tr_idx, va_idx in skf.split(X, y):
        model = Perceptron(X.shape[1], activation, LEARNING_RATE, MAX_EPOCHS, RANDOM_SEED)
        model.fit(X[tr_idx], y[tr_idx])

        train_accs.append(np.mean(model.predict(X[tr_idx]) == y[tr_idx]))
        val_accs.append(np.mean(model.predict(X[va_idx]) == y[va_idx]))
        epochs_used.append(model.epochs_run)
        n_converged += int(model.converged)

    return {
        "train_acc": np.mean(train_accs),
        "val_acc": np.mean(val_accs),
        "val_std": np.std(val_accs),
        "epochs": np.mean(epochs_used),
        "converged": n_converged,
    }


# -----------------------------------------------------------------------
# 5. Run everything
# -----------------------------------------------------------------------
def main():
    X, y = load_train()
    print(f"Training data: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Learning rate = {LEARNING_RATE}, max epochs = {MAX_EPOCHS}, "
          f"{N_FOLDS}-fold stratified CV\n")

    header = (f"{'activation':<10} {'train acc':>10} {'val acc':>9} {'val std':>8} "
              f"{'avg epochs':>11} {'folds converged':>16}")
    print(header)
    print("-" * len(header))

    for name in ACTIVATIONS:
        r = cross_validate(X, y, name)
        print(f"{name:<10} {r['train_acc']:>10.3f} {r['val_acc']:>9.3f} "
              f"{r['val_std']:>8.3f} {r['epochs']:>11.1f} "
              f"{r['converged']:>10d} / {N_FOLDS}")

        settings = f"activation={name};lr={LEARNING_RATE};max_epochs={MAX_EPOCHS};folds={N_FOLDS}"
        model_name = f"perceptron_{name}"
        log_result("perceptron.py", model_name, settings, RANDOM_SEED, "cv_train_acc", r["train_acc"])
        log_result("perceptron.py", model_name, settings, RANDOM_SEED, "cv_val_acc", r["val_acc"])
        log_result("perceptron.py", model_name, settings, RANDOM_SEED, "cv_val_acc_std", r["val_std"])
        log_result("perceptron.py", model_name, settings, RANDOM_SEED, "avg_epochs_run", r["epochs"])
        log_result("perceptron.py", model_name, settings, RANDOM_SEED, "folds_converged", r["converged"])

    # ---- Convergence plot: fit on the full training set, track errors ----
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name in ACTIVATIONS:
        model = Perceptron(X.shape[1], name, LEARNING_RATE, MAX_EPOCHS, RANDOM_SEED)
        model.fit(X, y)
        plt.plot(range(1, len(model.errors_per_epoch) + 1),
                 model.errors_per_epoch, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Misclassified training patients")
    plt.title("Perceptron convergence by activation function")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "perceptron_convergence.png"), dpi=150)
    print(f"\nSaved plot -> {FIG_DIR}/perceptron_convergence.png")
    print("Saved metrics -> experiments/results.csv")


if __name__ == "__main__":
    main()