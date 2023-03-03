import abc
from typing import Any, List, Tuple


class Params(abc.ABC):
    """
    Helper for describing the hyperparameters for each estimator.
    """

    def __init__(self, name: str, bounds: Tuple[Any, Any]) -> None:
        self.name = name
        self.bounds = bounds

    @abc.abstractmethod
    def get(self) -> List[Any]:
        ...

    @abc.abstractmethod
    def sample(self, trial: Any) -> Any:
        ...
