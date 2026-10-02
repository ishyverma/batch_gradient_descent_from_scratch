
import numpy as np


class BatchGradientDescent:
    """
    Batch Gradient Descent for Linear Regression.

    Parameters
    ----------
    learning_rate : float
        Step size for parameter updates.

    n_iterations : int
        Number of full-dataset gradient updates.
    """

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights = None
        self.bias = None
        self.loss_history = []

    def predict(self, X):
        return X @ self.weights + self.bias

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):

            y_pred = self.predict(X)

            errors = y_pred - y

            dw = -(2 / n_samples) * (X.T @ errors)
            db = -(2 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            updated_predictions = self.predict(X)

            loss = np.mean((updated_predictions - y) ** 2)

            self.loss_history.append(loss)

        return self
