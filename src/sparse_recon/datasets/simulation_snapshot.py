from pathlib import Path

import numpy as np

from sparse_recon.datasets.base import BaseDataset
from sparse_recon.types import FieldSnapshot


class SimulationSnapshotDataset(BaseDataset):
    name = "simulation_snapshot"

    def __init__(self, path: str | Path, loader_kwargs: dict | None = None):
        self.path = Path(path)
        self.loader_kwargs = loader_kwargs or {}

    def load(self) -> FieldSnapshot:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.path}\n"
                "Place the file locally or provide a valid simulation snapshot path."
            )
        if self.path.suffix.lower() != ".npy":
            raise ValueError(
                f"Unsupported simulation snapshot format: {self.path.suffix}. Expected .npy"
            )

        field = np.load(self.path, allow_pickle=False)
        if field.ndim != 4:
            raise ValueError(
                "Simulation snapshot must be a 4D array shaped (nx, ny, nz, 3); "
                f"got shape {field.shape}"
            )
        if field.shape[-1] != 3:
            raise ValueError(
                "Simulation snapshot last dimension must have size 3 for vector components; "
                f"got shape {field.shape}"
            )

        nx, ny, nz, n_components = field.shape
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(0.0, 1.0, ny)
        z = np.linspace(0.0, 1.0, nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        values = field.reshape(nx * ny * nz, n_components)

        metadata = {
            "source": "simulation",
            "field_kind": "simulation_snapshot",
            "file_path": str(self.path),
            "array_shape": list(field.shape),
            "dtype": str(field.dtype),
            "grid_convention": "uniform_unit_box",
        }

        return FieldSnapshot(
            coords=coords,
            values=values,
            grid_shape=(nx, ny, nz),
            axes={"x": x, "y": y, "z": z},
            metadata=metadata,
        )
