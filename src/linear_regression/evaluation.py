
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_regression(y_true, y_pred):
    """
    Calculate standard regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Actual target values.

    y_pred : array-like
        Predicted target values.

    Returns
    -------
    dict
        MAE, MSE, RMSE, and R2 scores.
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Actual and predicted values must have the same shape."
        )

    if not np.isfinite(y_true).all():
        raise ValueError("Actual values contain NaN or infinity.")

    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain NaN or infinity.")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
