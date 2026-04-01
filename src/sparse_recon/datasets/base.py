from abc import ABC, abstractmethod
from sparse_recon.types import FieldSnapshot


class BaseDataset(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> FieldSnapshot:
        raise NotImplementedError
