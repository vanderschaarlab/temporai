"""Module defining `Params` classes used for sampling in hyperparameter tuning."""

import abc
import random
from typing import Any, Generator, List, Optional, Tuple

import rich.pretty
from optuna.trial import Trial

RESERVED_ARG_NAMES = ("trail", "override")


class Params(abc.ABC):
    def __init__(self, name: str, bounds: Tuple[Any, Any]) -> None:
        """Abstract base class for all hyperparameter sampling classes. A helper for describing the hyperparameters for
        each estimator.

        Args:
            name (str): Hyperparameter name.
            bounds (Tuple[Any, Any]): The bounds (lower, higher) of the hyperparameter.
        """
        if name in RESERVED_ARG_NAMES:
            raise ValueError(f"The hyperparameter name '{name}' is not allowed, as it is a special argument")
        self.name = name
        self.bounds = bounds

    @abc.abstractmethod
    def get(self) -> List[Any]:  # pragma: no cover
        """Returns the hyperparameter name and properties as a list.

        Returns:
            List[Any]: The hyperparameter name and properties as a list.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def sample(self, trial: Optional[Trial] = None) -> Any:
        """Sample the hyperparameter. If `trial` is not `None`, dispatch to ``_sample_optuna_trial``. Otherwise,
        dispatch to ``_sample_default``.

        Args:
            trial (Optional[Trial], optional): Trial object, e.g `optuna.trial`. Defaults to None.

        Returns:
            Any: The sampled hyperparameter.
        """
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

    def __rich_repr__(self) -> Generator:
        """A `rich` representation of the class.

        Yields:
            Generator: The fields and their values fed to `rich`.
        """
        yield "name", self.name
        yield "bounds", self.bounds

    def __repr__(self) -> str:
        """The `repr()` representation of the class.

        Returns:
            str: The representation.
        """
        return rich.pretty.pretty_repr(self)


class CategoricalParams(Params):
    def __init__(self, name: str, choices: List[Any]) -> None:
        """Sample from a categorical distribution.

        Args:
            name (str): Hyperparameter name.
            choices (List[Any]): The choices to sample from.
        """
        super().__init__(name, (min(choices), max(choices)))
        self.name = name
        self.choices = choices

    def get(self) -> List[Any]:  # noqa: D102
        return [self.name, self.choices]

    def _sample_optuna_trial(self, trial: Trial) -> Any:
        return trial.suggest_categorical(self.name, self.choices)

    def _sample_default(self) -> Any:
        return random.SystemRandom().choice(self.choices)

    def __rich_repr__(self) -> Generator:
        """A `rich` representation of the class.

        Yields:
            Generator: The fields and their values fed to `rich`.
        """
        yield "name", self.name
        yield "choices", self.choices


class FloatParams(Params):
    def __init__(self, name: str, low: float, high: float) -> None:
        """Sample from a float distribution.

        Args:
            name (str): Hyperparameter name.
            low (float): Lower bound.
            high (float): Upper bound.
        """
        low = float(low)
        high = float(high)

        super().__init__(name, (low, high))
        self.name = name
        self.low = low
        self.high = high

    def get(self) -> List[Any]:  # noqa: D102
        return [self.name, self.low, self.high]

    def _sample_optuna_trial(self, trial: Trial) -> Any:
        return trial.suggest_float(self.name, self.low, self.high)

    def _sample_default(self) -> Any:
        return random.uniform(self.low, self.high)  # nosec

    def __rich_repr__(self) -> Generator:
        """A `rich` representation of the class.

        Yields:
            Generator: The fields and their values fed to `rich`.
        """
        yield "name", self.name
        yield "low", self.low
        yield "high", self.high


class IntegerParams(Params):
    def __init__(self, name: str, low: int, high: int, step: int = 1) -> None:
        """Sample from an integer distribution.

        Args:
            name (str): Hyperparameter name.
            low (int): Lower bound.
            high (int): Upper bound.
            step (int, optional): Step. Defaults to ``1``.
        """
        self.low = low
        self.high = high
        self.step = step

        super().__init__(name, (low, high))
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.choices = [val for val in range(low, high + 1, step)]

    def get(self) -> List[Any]:  # noqa: D102
        return [self.name, self.low, self.high, self.step]

    def _sample_optuna_trial(self, trial: Trial) -> Any:
        return trial.suggest_int(self.name, self.low, self.high, self.step)

    def _sample_default(self) -> Any:
        return random.SystemRandom().choice(self.choices)

    def __rich_repr__(self) -> Generator:
        """A `rich` representation of the class.

        Yields:
            Generator: The fields and their values fed to `rich`.
        """
        yield "name", self.name
        yield "low", self.low
        yield "high", self.high
        yield "step", self.step
