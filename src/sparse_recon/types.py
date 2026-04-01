from dataclasses import dataclass
import numpy as np

Array = np.ndarray


@dataclass
class FieldSnapshot:
    """
    Represents one vector field snapshot on a structured or unstructured domain.
    """
    coords: Array          # shape (N, d)
    values: Array          # shape (N, c)
    grid_shape: tuple | None = None
    axes: dict | None = None
    metadata: dict | None = None


@dataclass
class SampleSet:
    """
    Sparse point measurements taken from the field.
    """
    coords: Array          # shape (M, d)
    values: Array          # shape (M, c)
    metadata: dict | None = None


@dataclass
class ReconstructionResult:
    method: str
    query_coords: Array
    predicted_values: Array
    metrics: dict
    metadata: dict | None = None
