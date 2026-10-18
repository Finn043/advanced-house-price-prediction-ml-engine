import numpy as np


def loss_and_gradient(y_true, y_pred, name="mse", delta=1.0):
    """Return mean loss and its gradient with respect to predictions."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must have the same non-empty shape")
    error = y_pred - y_true
    count = error.size
    if name == "mse":
        return float(np.mean(error ** 2)), 2 * error / count
    if name == "mae":
        return float(np.mean(np.abs(error))), np.sign(error) / count
    if name == "huber":
        if delta <= 0:
            raise ValueError("delta must be positive")
        absolute = np.abs(error)
        values = np.where(absolute <= delta, 0.5 * error ** 2, delta * (absolute - 0.5 * delta))
        return float(np.mean(values)), np.clip(error, -delta, delta) / count
    raise ValueError("loss must be 'mse', 'mae', or 'huber'")
