"""Regression metric computations."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%). Ignores zero actuals."""
    y_true, y_pred = _align(y_true, y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error (%)."""
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (%)."""
    y_true, y_pred = _align(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def bias(y_true, y_pred):
    """Mean bias (positive = over-prediction)."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def compute_metrics(y_true, y_pred):
    """Return all regression metrics as a dict.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Metric name -> value mapping.
    """
    y_true, y_pred = _align(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def _align(y_true, y_pred):
    """Ensure inputs are numpy arrays of the same length."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return y_true, y_pred
