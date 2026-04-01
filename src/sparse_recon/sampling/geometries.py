import numpy as np


def random_points_in_box(
    n_points: int,
    dim: int,
    low=0.0,
    high=1.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_points, dim))


def clustered_points(center, offsets) -> np.ndarray:
    center = np.asarray(center)
    offsets = np.asarray(offsets)
    return center[None, :] + offsets


def tetrahedron_like(scale: float = 0.1, center=(0.5, 0.5, 0.5)) -> np.ndarray:
    center = np.asarray(center)
    offsets = scale * np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=float,
    )
    return center[None, :] + offsets


def clustered_points_in_box(
    n_points: int,
    dim: int,
    seed: int = 0,
    low: float = 0.0,
    high: float = 1.0,
    n_clusters: int = 3,
    cluster_scale: float = 0.08,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(low + 0.15, high - 0.15, size=(n_clusters, dim))
    cluster_ids = rng.integers(0, n_clusters, size=n_points)
    offsets = rng.normal(scale=cluster_scale, size=(n_points, dim))
    points = centers[cluster_ids] + offsets
    return np.clip(points, low, high)


def multi_probe_like_points_2d(
    n_points: int,
    seed: int = 0,
    low: float = 0.0,
    high: float = 1.0,
    spacing: float = 0.045,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
        ]
    )
    offsets = spacing * np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )

    points = []
    while len(points) < n_points:
        for center in centers:
            jitter = rng.normal(scale=spacing / 5.0, size=offsets.shape)
            probe = np.clip(center + offsets + jitter, low, high)
            points.extend(probe.tolist())
            if len(points) >= n_points:
                break
    return np.asarray(points[:n_points], dtype=float)


def generate_sampling_points(
    geometry: str,
    n_points: int,
    dim: int,
    seed: int = 0,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    if geometry == "random":
        return random_points_in_box(n_points=n_points, dim=dim, low=low, high=high, seed=seed)
    if geometry == "clustered":
        return clustered_points_in_box(
            n_points=n_points,
            dim=dim,
            seed=seed,
            low=low,
            high=high,
        )
    if geometry == "multi_probe_like":
        if dim != 2:
            raise ValueError("multi_probe_like geometry currently supports dim=2 only")
        return multi_probe_like_points_2d(
            n_points=n_points,
            seed=seed,
            low=low,
            high=high,
        )

    supported = ["clustered", "multi_probe_like", "random"]
    raise ValueError(f"Unknown geometry '{geometry}'. Supported: {', '.join(supported)}")
