import abc
import random
from typing import Any, List, Optional, Tuple

from optuna.trial import Trial


class Params(abc.ABC):
    """Helper for describing the hyperparameters for each estimator."""

    def __init__(self, name: str, bounds: Tuple[Any, Any]) -> None:
        self.name = name
        self.bounds = bounds

    @abc.abstractmethod
    def get(self) -> List[Any]:  # pragma: no cover
        ...

    def sample(self, trial: Optional[Trial] = None) -> Any:
        if trial is not None:
            return self._sample_optuna_trial(trial)
        else:
            return self._sample_default()
        # NOTE: Could support more parameter sampling implementations.

    @abc.abstractmethod
    def _sample_optuna_trial(self, trial: Trial) -> Any:  # pragma: no cover
        ...

    @abc.abstractmethod
    def _sample_default(self) -> Any:  # pragma: no cover
        ...


class CategoricalParams(Params):
    """Sample from a categorical distribution."""

    def __init__(self, name: str, choices: List[Any]) -> None:
        super().__init__(name, (min(choices), max(choices)))
        self.name = name
        self.choices = choices

    def get(self) -> List[Any]:
        return [self.name, self.choices]

    def _sample_optuna_trial(self, trial: Trial) -> Any:
        return trial.suggest_categorical(self.name, self.choices)

    def _sample_default(self) -> Any:
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

    def _sample_optuna_trial(self, trial: Trial) -> Any:
        return trial.suggest_float(self.name, self.low, self.high)

    def _sample_default(self) -> Any:
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

    def _sample_optuna_trial(self, trial: Trial) -> Any:
        return trial.suggest_int(self.name, self.low, self.high, self.step)

    def _sample_default(self) -> Any:
        return random.SystemRandom().choice(self.choices)
