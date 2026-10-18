import numpy as np

from model.gradient_descent import optimize


class LinearRegression:
    """Linear regression fitted with the project's NumPy gradient descent."""
    def fit(self, X, y, **optimizer_options):
        self.weights, self.bias, self.history = optimize(X, y, **optimizer_options)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.weights.size:
            raise ValueError("X must be 2D and match the number of fitted features")
        return X @ self.weights + self.bias
