import numpy as np

from sparse_recon.types import FieldSnapshot


def _make_grid_2d(
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return x, y, xx, yy


def _make_grid_3d(
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.linspace(0.0, 1.0, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, xx, yy, zz


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
    x, y, xx, yy = _make_grid_2d(nx, ny)

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
    x, y, xx, yy = _make_grid_2d(nx, ny)

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


def _snapshot_from_vector_potential(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    metadata: dict | None = None,
) -> FieldSnapshot:
    d_az_dy = np.gradient(az, y, axis=1)
    d_ay_dz = np.gradient(ay, z, axis=2)
    d_ax_dz = np.gradient(ax, z, axis=2)
    d_az_dx = np.gradient(az, x, axis=0)
    d_ay_dx = np.gradient(ay, x, axis=0)
    d_ax_dy = np.gradient(ax, y, axis=1)

    bx = d_az_dy - d_ay_dz
    by = d_ax_dz - d_az_dx
    bz = d_ay_dx - d_ax_dy

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])

    return FieldSnapshot(
        coords=coords,
        values=values,
        grid_shape=(len(x), len(y), len(z)),
        axes={"x": x, "y": y, "z": z},
        metadata=metadata or {},
    )


def make_smooth_3d_vector_field(
    nx: int = 24,
    ny: int = 24,
    nz: int = 24,
    seed: int = 0,
) -> FieldSnapshot:
    rng = np.random.default_rng(seed)
    x, y, z, xx, yy, zz = _make_grid_3d(nx, ny, nz)

    phase = rng.uniform(-0.5, 0.5, size=3)
    ax = 0.35 * np.cos(2 * np.pi * yy + phase[0]) * np.sin(2 * np.pi * zz)
    ay = 0.25 * np.sin(2 * np.pi * xx + phase[1]) * np.cos(2 * np.pi * zz)
    az = (
        np.sin(2 * np.pi * xx + phase[2]) * np.cos(2 * np.pi * yy)
        + 0.2 * np.cos(4 * np.pi * zz - phase[0])
    )

    return _snapshot_from_vector_potential(
        ax,
        ay,
        az,
        x,
        y,
        z,
        metadata={
            "source": "synthetic",
            "field_kind": "smooth_3d",
            "dim": 3,
            "divergence_free": True,
            "seed": seed,
        },
    )


def make_high_frequency_3d_vector_field(
    nx: int = 24,
    ny: int = 24,
    nz: int = 24,
    seed: int = 0,
) -> FieldSnapshot:
    rng = np.random.default_rng(seed)
    x, y, z, xx, yy, zz = _make_grid_3d(nx, ny, nz)

    phase = rng.uniform(-np.pi, np.pi, size=4)
    ax = (
        0.3 * np.sin(6 * np.pi * yy + phase[0]) * np.cos(5 * np.pi * zz + phase[1])
        + 0.1 * np.cos(8 * np.pi * (yy + zz))
    )
    ay = (
        0.3 * np.cos(6 * np.pi * xx + phase[2]) * np.sin(5 * np.pi * zz)
        + 0.12 * np.sin(9 * np.pi * (xx - zz))
    )
    az = (
        0.9 * np.sin(5 * np.pi * xx + phase[3]) * np.cos(4 * np.pi * yy)
        + 0.2 * np.sin(7 * np.pi * (xx + yy + zz))
    )

    return _snapshot_from_vector_potential(
        ax,
        ay,
        az,
        x,
        y,
        z,
        metadata={
            "source": "synthetic",
            "field_kind": "high_frequency_3d",
            "dim": 3,
            "divergence_free": True,
            "seed": seed,
        },
    )


def create_synthetic_field(
    kind: str,
    nx: int = 64,
    ny: int = 64,
    nz: int | None = None,
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
        "smooth_3d": lambda: make_smooth_3d_vector_field(
            nx=nx,
            ny=ny,
            nz=nz or nx,
            seed=seed,
        ),
        "high_frequency_3d": lambda: make_high_frequency_3d_vector_field(
            nx=nx,
            ny=ny,
            nz=nz or nx,
            seed=seed,
        ),
    }

    try:
        return builders[kind]()
    except KeyError as exc:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unknown synthetic field kind '{kind}'. Supported: {supported}") from exc
