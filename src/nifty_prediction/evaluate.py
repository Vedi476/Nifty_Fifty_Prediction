import numpy as np


def evaluate(y_true, y_pred):
    return {
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "MSE": np.mean((y_true - y_pred) ** 2),
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "R2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        "Correlation": np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1],
    }
