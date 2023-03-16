import abc
import random
from typing import Any, List, Tuple


class Params(abc.ABC):
    """Helper for describing the hyperparameters for each estimator."""

    def __init__(self, name: str, bounds: Tuple[Any, Any]) -> None:
        self.name = name
        self.bounds = bounds

    @abc.abstractmethod
    def get(self) -> List[Any]:
        ...

    @abc.abstractmethod
    def sample(self) -> Any:
        ...


class CategoricalParams(Params):
    """Sample from a categorical distribution."""

    def __init__(self, name: str, choices: List[Any]) -> None:
        super().__init__(name, (min(choices), max(choices)))
        self.name = name
        self.choices = choices

    def get(self) -> List[Any]:
        return [self.name, self.choices]

    def sample(self) -> Any:
        return random.SystemRandom().choice(self.choices)


class FloatParams(Params):
    """Sample from a float distribution."""

    def __init__(self, name: str, low: float, high: float) -> None:
        low = float(low)
        high = float(high)

        super().__init__(name, (low, high))
        self.name = name
        self.low = low
        self.high = high

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    def sample(self) -> Any:
        return random.uniform(self.low, self.high)  # nosec


class IntegerParams(Params):
    """Sample from an integer distribution."""

    def __init__(self, name: str, low: int, high: int, step: int = 1) -> None:
        self.low = low
        self.high = high
        self.step = step

        super().__init__(name, (low, high))
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.choices = [val for val in range(low, high + 1, step)]

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step]

    def sample(self) -> Any:
        return random.SystemRandom().choice(
            self.choices,
        )
