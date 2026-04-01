import numpy as np
from scipy.interpolate import RBFInterpolator
from sparse_recon.methods.base import ReconstructionMethod


class RBFMethod(ReconstructionMethod):
    name = "rbf"

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        epsilon: float | None = None,
        neighbors: int | None = None,
        smoothing: float = 0.0,
    ):
        self.kernel = kernel
        self.epsilon = epsilon
        self.neighbors = neighbors
        self.smoothing = smoothing

    def fit(self, sample_coords: np.ndarray, sample_values: np.ndarray):
        self.interp = RBFInterpolator(
            sample_coords,
            sample_values,
            kernel=self.kernel,
            epsilon=self.epsilon,
            neighbors=self.neighbors,
            smoothing=self.smoothing,
        )
        return self

    def predict(self, query_coords: np.ndarray) -> np.ndarray:
        return self.interp(query_coords)

    def get_params(self) -> dict:
        return {
            "kernel": self.kernel,
            "epsilon": self.epsilon,
            "neighbors": self.neighbors,
            "smoothing": self.smoothing,
        }
