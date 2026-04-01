import numpy as np


def divergence_2d(values: np.ndarray, grid_shape: tuple[int, int], x: np.ndarray, y: np.ndarray):
    bx = values[:, 0].reshape(grid_shape)
    by = values[:, 1].reshape(grid_shape)

    dbx_dx = np.gradient(bx, x, axis=0)
    dby_dy = np.gradient(by, y, axis=1)

    return dbx_dx + dby_dy


def divergence_rmse_2d(truth: np.ndarray, pred: np.ndarray, grid_shape, x, y) -> float:
    if not np.isfinite(pred).all():
        return np.nan
    div_truth = divergence_2d(truth, grid_shape, x, y)
    div_pred = divergence_2d(pred, grid_shape, x, y)
    return float(np.sqrt(np.mean((div_truth - div_pred) ** 2)))
