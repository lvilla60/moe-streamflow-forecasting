"""Small NumPy regression metrics."""

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(error ** 2)))


def nse(y_true, y_pred) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    denominator = np.sum((true - np.mean(true)) ** 2)
    if denominator == 0:
        raise ValueError("NSE is undefined when y_true has zero variance")
    return float(1 - np.sum((true - pred) ** 2) / denominator)
