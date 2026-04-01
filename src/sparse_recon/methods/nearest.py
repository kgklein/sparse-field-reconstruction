import numpy as np
from scipy.spatial import cKDTree
from sparse_recon.methods.base import ReconstructionMethod


class NearestMethod(ReconstructionMethod):
    name = "nearest"

    def fit(self, sample_coords: np.ndarray, sample_values: np.ndarray):
        self.tree = cKDTree(sample_coords)
        self.sample_values = sample_values
        return self

    def predict(self, query_coords: np.ndarray) -> np.ndarray:
        _, idx = self.tree.query(query_coords, k=1)
        return self.sample_values[idx]
