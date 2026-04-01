from abc import ABC, abstractmethod

import numpy as np


class ReconstructionMethod(ABC):
    name = "base"

    @abstractmethod
    def fit(self, sample_coords: np.ndarray, sample_values: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def predict(self, query_coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reconstruct(self, query_coords: np.ndarray) -> np.ndarray:
        return self.predict(query_coords)

    def get_params(self) -> dict:
        return {}
