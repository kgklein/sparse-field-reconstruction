import numpy as np


def isotropic_power_spectrum_2d(values: np.ndarray, grid_shape: tuple[int, int]):
    """
    Placeholder for a simple 2D isotropic spectrum estimate.
    """
    vx = values[:, 0].reshape(grid_shape)
    vy = values[:, 1].reshape(grid_shape)

    fx = np.fft.fftn(vx)
    fy = np.fft.fftn(vy)
    power = np.abs(fx) ** 2 + np.abs(fy) ** 2

    return power
