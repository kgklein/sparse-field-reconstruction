import numpy as np
from scipy.spatial import cKDTree

from sparse_recon.types import FieldSnapshot, SampleSet


def sample_field_nearest(field: FieldSnapshot, sample_coords: np.ndarray) -> SampleSet:
    tree = cKDTree(field.coords)
    _, idx = tree.query(sample_coords, k=1)
    values = field.values[idx]
    metadata = {
        "sampling": "nearest",
        "n_samples": int(len(sample_coords)),
    }
    return SampleSet(coords=sample_coords, values=values, metadata=metadata)


def add_noise(samples: SampleSet, sigma: float = 0.0, seed: int = 0) -> SampleSet:
    if sigma <= 0:
        return samples

    rng = np.random.default_rng(seed)
    noisy_values = samples.values + sigma * rng.standard_normal(samples.values.shape)
    metadata = dict(samples.metadata or {})
    metadata.update({"noise_sigma": sigma, "noise_seed": seed})
    return SampleSet(coords=samples.coords, values=noisy_values, metadata=metadata)


def sample_field(
    field: FieldSnapshot,
    sample_coords: np.ndarray,
    *,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> SampleSet:
    samples = sample_field_nearest(field, sample_coords)
    metadata = dict(samples.metadata or {})
    metadata.update(
        {
            "field_source": (field.metadata or {}).get("source", "unknown"),
            "field_kind": (field.metadata or {}).get("field_kind", "unknown"),
        }
    )
    samples = SampleSet(coords=samples.coords, values=samples.values, metadata=metadata)
    return add_noise(samples, sigma=noise_sigma, seed=seed)
