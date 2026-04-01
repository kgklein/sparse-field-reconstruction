import numpy as np
from scipy.interpolate import LinearNDInterpolator
from sparse_recon.methods.base import ReconstructionMethod


class LinearMethod(ReconstructionMethod):
    name = "linear"

    def fit(self, sample_coords: np.ndarray, sample_values: np.ndarray):
        self.interps = [
            LinearNDInterpolator(sample_coords, sample_values[:, i], fill_value=np.nan)
            for i in range(sample_values.shape[1])
        ]
        return self

    def predict(self, query_coords: np.ndarray) -> np.ndarray:
        pred = np.column_stack([interp(query_coords) for interp in self.interps])
        return pred
