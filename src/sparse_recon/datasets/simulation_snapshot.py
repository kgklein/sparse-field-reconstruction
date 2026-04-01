from pathlib import Path
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
                "Place the large file locally or mount it from Box."
            )

        # Placeholder:
        # This is where you adapt the loader logic from the older repo.
        # Return coords, values, grid_shape, axes, metadata.

        raise NotImplementedError(
            "Implement the snapshot reader using the legacy loader logic."
        )
