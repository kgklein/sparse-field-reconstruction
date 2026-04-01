import numpy as np


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(pred).all(axis=1)
    return float(np.sqrt(np.mean((truth[mask] - pred[mask]) ** 2)))


def relative_l2(truth: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(pred).all(axis=1)
    num = np.linalg.norm(truth[mask] - pred[mask])
    den = np.linalg.norm(truth[mask])
    return float(num / den) if den > 0 else np.nan
