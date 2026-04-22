from pathlib import Path

import numpy as np

from sparse_recon.datasets.base import BaseDataset
from sparse_recon.datasets.structured_snapshot import load_structured_snapshot_data
from sparse_recon.types import FieldSnapshot


class SimulationSnapshotDataset(BaseDataset):
    name = "simulation_snapshot"

    def __init__(self, path: str | Path, loader_kwargs: dict | None = None):
        self.path = Path(path)
        self.loader_kwargs = loader_kwargs or {}

    def load(self) -> FieldSnapshot:
        field, axes, grid_shape, metadata = load_structured_snapshot_data(
            self.path,
            **self.loader_kwargs,
        )

        nx, ny, nz = grid_shape
        n_components = field.shape[-1]
        x = np.asarray(axes["x"], dtype=float)
        y = np.asarray(axes["y"], dtype=float)
        z = np.asarray(axes["z"], dtype=float)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        values = field.reshape(nx * ny * nz, n_components)

        return FieldSnapshot(
            coords=coords,
            values=values,
            grid_shape=grid_shape,
            axes={"x": x, "y": y, "z": z},
            metadata=metadata,
        )
