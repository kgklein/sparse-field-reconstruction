import numpy as np
from sparse_recon.types import FieldSnapshot


def make_smooth_2d_vector_field(nx: int = 64, ny: int = 64, seed: int = 0) -> FieldSnapshot:
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Simple stream-function style field for a divergence-free 2D example
    psi = (
        np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
        + 0.25 * np.sin(4 * np.pi * xx + 0.3) * np.cos(3 * np.pi * yy - 0.2)
    )

    bx = np.gradient(psi, y, axis=1)
    by = -np.gradient(psi, x, axis=0)

    coords = np.column_stack([xx.ravel(), yy.ravel()])
    values = np.column_stack([bx.ravel(), by.ravel()])

    return FieldSnapshot(
        coords=coords,
        values=values,
        grid_shape=(nx, ny),
        axes={"x": x, "y": y},
        metadata={"source": "synthetic", "divergence_free": True},
    )
