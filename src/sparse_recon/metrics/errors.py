import numpy as np


def finite_prediction_mask(pred: np.ndarray) -> np.ndarray:
    return np.isfinite(pred).all(axis=1)


def valid_fraction(pred: np.ndarray) -> float:
    mask = finite_prediction_mask(pred)
    return float(mask.mean()) if len(mask) else np.nan


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    mask = finite_prediction_mask(pred)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((truth[mask] - pred[mask]) ** 2)))


def relative_l2(truth: np.ndarray, pred: np.ndarray) -> float:
    mask = finite_prediction_mask(pred)
    if not np.any(mask):
        return np.nan
    num = np.linalg.norm(truth[mask] - pred[mask])
    den = np.linalg.norm(truth[mask])
    return float(num / den) if den > 0 else np.nan


def prediction_validity_summary(pred: np.ndarray) -> dict:
    mask = finite_prediction_mask(pred)
    valid_count = int(mask.sum())
    total_count = int(len(mask))
    invalid_count = total_count - valid_count
    return {
        "valid_fraction": float(valid_count / total_count) if total_count else np.nan,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
    }
