import numpy as np
from scipy.spatial import cKDTree
from sparse_recon.types import FieldSnapshot, SampleSet


def sample_field_nearest(field: FieldSnapshot, sample_coords: np.ndarray) -> SampleSet:
    tree = cKDTree(field.coords)
    _, idx = tree.query(sample_coords, k=1)
    values = field.values[idx]
    return SampleSet(coords=sample_coords, values=values, metadata={"sampling": "nearest"})


def add_noise(samples: SampleSet, sigma: float = 0.0, seed: int = 0) -> SampleSet:
    if sigma <= 0:
        return samples

    rng = np.random.default_rng(seed)
    noisy_values = samples.values + sigma * rng.standard_normal(samples.values.shape)
    return SampleSet(coords=samples.coords, values=noisy_values, metadata={"noise_sigma": sigma})
