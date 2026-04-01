import numpy as np


def random_points_in_box(n_points: int, dim: int, low=0.0, high=1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_points, dim))


def clustered_points(center, offsets):
    center = np.asarray(center)
    offsets = np.asarray(offsets)
    return center[None, :] + offsets


def tetrahedron_like(scale: float = 0.1, center=(0.5, 0.5, 0.5)):
    center = np.asarray(center)
    offsets = scale * np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float)
    return center[None, :] + offsets
