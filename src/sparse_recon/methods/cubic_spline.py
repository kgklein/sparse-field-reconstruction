import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

from sparse_recon.methods.base import ReconstructionMethod


class CubicSplineMethod(ReconstructionMethod):
    """Cubic spline interpolation using Clough-Tocher scheme (C1 smooth, 2D only).

    Falls back to linear interpolation outside the convex hull of sample points,
    matching the behaviour of scipy's LinearNDInterpolator for extrapolation.
    """

    name = "cubic_spline"

    def __init__(self, fill_value: float = np.nan):
        self.fill_value = fill_value

    def fit(self, sample_coords: np.ndarray, sample_values: np.ndarray):
        self._cubic = CloughTocher2DInterpolator(
            sample_coords,
            sample_values,
            fill_value=self.fill_value,
        )
        self._linear_fallback = LinearNDInterpolator(
            sample_coords,
            sample_values,
            fill_value=self.fill_value,
        )
        return self

    def predict(self, query_coords: np.ndarray) -> np.ndarray:
        pred = self._cubic(query_coords)
        # where cubic returns nan (outside convex hull), use linear fallback
        nan_mask = np.any(np.isnan(pred), axis=-1)
        if np.any(nan_mask):
            pred[nan_mask] = self._linear_fallback(query_coords[nan_mask])
        return pred

    def get_params(self) -> dict:
        return {"fill_value": self.fill_value}
