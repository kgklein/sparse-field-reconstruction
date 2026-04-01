import numpy as np

from sparse_recon.types import FieldSnapshot


def _make_grid(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return x, y, xx, yy


def _snapshot_from_stream_function(
    psi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    metadata: dict | None = None,
) -> FieldSnapshot:
    bx = np.gradient(psi, y, axis=1)
    by = -np.gradient(psi, x, axis=0)

    xx, yy = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    values = np.column_stack([bx.ravel(), by.ravel()])

    return FieldSnapshot(
        coords=coords,
        values=values,
        grid_shape=(len(x), len(y)),
        axes={"x": x, "y": y},
        metadata=metadata or {},
    )


def make_smooth_2d_vector_field(
    nx: int = 64,
    ny: int = 64,
    seed: int = 0,
) -> FieldSnapshot:
    rng = np.random.default_rng(seed)
    x, y, xx, yy = _make_grid(nx, ny)

    phase_x, phase_y = rng.uniform(-0.5, 0.5, size=2)
    psi = (
        np.sin(2 * np.pi * xx + phase_x) * np.cos(2 * np.pi * yy + phase_y)
        + 0.25 * np.sin(4 * np.pi * xx + 0.3) * np.cos(3 * np.pi * yy - 0.2)
    )

    return _snapshot_from_stream_function(
        psi,
        x,
        y,
        metadata={
            "source": "synthetic",
            "field_kind": "smooth",
            "divergence_free": True,
            "seed": seed,
        },
    )


def make_high_frequency_2d_vector_field(
    nx: int = 64,
    ny: int = 64,
    seed: int = 0,
) -> FieldSnapshot:
    rng = np.random.default_rng(seed)
    x, y, xx, yy = _make_grid(nx, ny)

    phase = rng.uniform(-np.pi, np.pi, size=3)
    psi = (
        0.8 * np.sin(6 * np.pi * xx + phase[0]) * np.cos(5 * np.pi * yy + phase[1])
        + 0.35 * np.sin(10 * np.pi * (xx + yy) + phase[2])
        + 0.15 * np.cos(12 * np.pi * xx - 7 * np.pi * yy)
    )

    return _snapshot_from_stream_function(
        psi,
        x,
        y,
        metadata={
            "source": "synthetic",
            "field_kind": "high_frequency",
            "divergence_free": True,
            "seed": seed,
        },
    )


def make_noisy_2d_vector_field(
    nx: int = 64,
    ny: int = 64,
    seed: int = 0,
    noise_sigma: float = 0.05,
) -> FieldSnapshot:
    base = make_smooth_2d_vector_field(nx=nx, ny=ny, seed=seed)
    rng = np.random.default_rng(seed + 1)
    noisy_values = base.values + noise_sigma * rng.standard_normal(base.values.shape)

    metadata = dict(base.metadata or {})
    metadata.update(
        {
            "field_kind": "noisy_smooth",
            "noise_sigma": noise_sigma,
        }
    )

    return FieldSnapshot(
        coords=base.coords,
        values=noisy_values,
        grid_shape=base.grid_shape,
        axes=base.axes,
        metadata=metadata,
    )


def create_synthetic_field(
    kind: str,
    nx: int = 64,
    ny: int = 64,
    seed: int = 0,
    noise_sigma: float = 0.05,
) -> FieldSnapshot:
    builders = {
        "smooth": lambda: make_smooth_2d_vector_field(nx=nx, ny=ny, seed=seed),
        "high_frequency": lambda: make_high_frequency_2d_vector_field(
            nx=nx,
            ny=ny,
            seed=seed,
        ),
        "noisy_smooth": lambda: make_noisy_2d_vector_field(
            nx=nx,
            ny=ny,
            seed=seed,
            noise_sigma=noise_sigma,
        ),
    }

    try:
        return builders[kind]()
    except KeyError as exc:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unknown synthetic field kind '{kind}'. Supported: {supported}") from exc
