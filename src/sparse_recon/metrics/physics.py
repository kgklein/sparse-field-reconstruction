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


def divergence_3d(
    values: np.ndarray,
    grid_shape: tuple[int, int, int],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
):
    bx = values[:, 0].reshape(grid_shape)
    by = values[:, 1].reshape(grid_shape)
    bz = values[:, 2].reshape(grid_shape)

    dbx_dx = np.gradient(bx, x, axis=0)
    dby_dy = np.gradient(by, y, axis=1)
    dbz_dz = np.gradient(bz, z, axis=2)

    return dbx_dx + dby_dy + dbz_dz


def divergence_rmse_3d(
    truth: np.ndarray,
    pred: np.ndarray,
    grid_shape,
    x,
    y,
    z,
) -> float:
    if not np.isfinite(pred).all():
        return np.nan
    div_truth = divergence_3d(truth, grid_shape, x, y, z)
    div_pred = divergence_3d(pred, grid_shape, x, y, z)
    return float(np.sqrt(np.mean((div_truth - div_pred) ** 2)))
