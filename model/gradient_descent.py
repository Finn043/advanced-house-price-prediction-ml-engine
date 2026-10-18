import numpy as np

from model.loss_functions import loss_and_gradient


def optimize(X, y, learning_rate=0.03, epochs=500, method="batch", loss="mse", delta=1.0, seed=42):
    """Optimize linear-regression weights using batch GD or single-row SGD."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
        raise ValueError("X must be 2D and have the same non-zero row count as y")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    if learning_rate <= 0 or epochs < 1:
        raise ValueError("learning_rate and epochs must be positive")
    if method not in ("batch", "sgd"):
        raise ValueError("method must be 'batch' or 'sgd'")

    weights, bias = np.zeros(X.shape[1]), 0.0
    history, rng = [], np.random.default_rng(seed)
    for _ in range(epochs):
        if method == "batch":
            _, gradient = loss_and_gradient(y, X @ weights + bias, loss, delta)
            weights -= learning_rate * (X.T @ gradient)
            bias -= learning_rate * float(gradient.sum())
        else:
            for index in rng.permutation(y.size):
                prediction = np.array([X[index] @ weights + bias])
                _, gradient = loss_and_gradient(y[index:index + 1], prediction, loss, delta)
                weights -= learning_rate * (X[index] * gradient[0])
                bias -= learning_rate * float(gradient[0])
        history.append(loss_and_gradient(y, X @ weights + bias, loss, delta)[0])
    return weights, bias, history
