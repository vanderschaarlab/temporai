"""Module containing the base class for metrics."""

# pylint: disable=unnecessary-ellipsis

import abc
from typing import Any, Generator, List

import numpy as np
import rich.pretty

import tempor.core.utils
from tempor.core import plugins
from tempor.data import data_typing

from . import metric_typing


class Metric(plugins.Plugin, abc.ABC):
    """Metric abstract base class, defines the required methods."""

    @property
    @abc.abstractmethod
    def direction(self) -> metric_typing.MetricDirection:  # pragma: no cover
        """The direction of the metric"""
        ...

    def evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """The metric evaluation call.

        Args:
            actual (Any): Actual value(s).
            predicted (Any): Predicted value(s).
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: Evaluated metric value(s)
        """
        self._validate(actual, predicted)
        return self._evaluate(actual, predicted, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """A convenience method to call `evaluate` directly."""
        return self.evaluate(*args, **kwargs)

    @abc.abstractmethod
    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """The metric evaluation call *implementation* to be overridden in derived classes.

        Args:
            actual (Any): Actual value(s).
            predicted (Any): Predicted value(s).
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: Evaluated metric value(s).
        """
        ...

    def _validate(self, actual: Any, predicted: Any) -> None:
        """Input validation. Can be overridden in derived classes, but a call to `super()._validate(...)` should
        be preserved.

        Args:
            actual (Any): Actual value(s).
            predicted (Any): Predicted value(s).
        """
        if actual is None or predicted is None:
            raise ValueError("The actual values and the predicted values must not be `None`.")

    def __rich_repr__(self) -> Generator:
        """A `rich` representation of the class. The ``description`` field is auto-generated from the class and init
        docstrings.

        Yields:
            Generator: The fields and their values fed to `rich`.
        """
        yield "name", self.name
        yield "description", tempor.core.utils.make_description_from_doc(self)

    def __repr__(self) -> str:
        """The `repr()` representation of the class.

        Returns:
            str: The representation.
        """
        return rich.pretty.pretty_repr(self)


# TODO: Multi-feature cases.
# TODO: Typing of arguments may change.
# TODO: Update the abstract methods for each case properly.


class OneOffPredictionMetric(Metric):
    """Metric abstract base class for the one-off prediction task."""

    # Overridden for type hinting.
    @abc.abstractmethod
    def _evaluate(
        self, actual: np.ndarray, predicted: np.ndarray, *args: Any, **kwargs: Any
    ) -> float:  # pragma: no cover # noqa: D102
        ...


# Overrides for type hinting and docstrings.
class OneOffClassificationMetric(OneOffPredictionMetric):
    """Metric abstract base class for the one-off prediction task, classification case."""

    def evaluate(self, actual: np.ndarray, predicted: np.ndarray, *args: Any, **kwargs: Any) -> float:
        """The metric evaluation call.

        ``actual`` and ``predicted`` are expected to be numpy arrays (sample in the 0th dimension).

        ``predicted`` must be the probabilities in this case.

        Args:
            actual (np.ndarray): Actual value(s).
            predicted (np.ndarray): Predicted value(s).
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            float: Evaluated metric value.
        """
        return super().evaluate(actual, predicted, *args, **kwargs)

    @abc.abstractmethod
    def _evaluate(
        self, actual: np.ndarray, predicted: np.ndarray, *args: Any, **kwargs: Any
    ) -> float:  # pragma: no cover # noqa: D102
        ...


plugins.register_plugin_category("prediction.one_off.classification", OneOffClassificationMetric, plugin_type="metric")


# Overrides for type hinting and docstrings.
class OneOffRegressionMetric(OneOffPredictionMetric):
    """Metric abstract base class for the one-off prediction task, regression case."""

    def evaluate(self, actual: np.ndarray, predicted: np.ndarray, *args: Any, **kwargs: Any) -> float:
        """The metric evaluation call.

        ``actual`` and ``predicted`` are expected to be numpy arrays (sample in the 0th dimension).

        Args:
            actual (np.ndarray): Actual value(s).
            predicted (np.ndarray): Predicted value(s).
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            float: Evaluated metric value.
        """
        return super().evaluate(actual, predicted, *args, **kwargs)

    @abc.abstractmethod
    def _evaluate(
        self, actual: np.ndarray, predicted: np.ndarray, *args: Any, **kwargs: Any
    ) -> float:  # pragma: no cover # noqa: D102
        ...


plugins.register_plugin_category("prediction.one_off.regression", OneOffRegressionMetric, plugin_type="metric")


# Overrides for type hinting and docstrings.
class TimeToEventMetric(Metric):
    """Metric abstract base class for the time-to-event (survival) task."""

    def evaluate(  # pylint: disable=arguments-differ
        self,
        actual: metric_typing.EventArrayTimeArray,
        predicted: np.ndarray,
        horizons: data_typing.TimeIndex,
        *args: Any,
        **kwargs: Any,
    ) -> List[float]:
        """The metric evaluation call.

        Args:
            actual (metric_typing.EventArrayTimeArray):
                A tuple of two numpy arrays: the event values array and the event times array,
                for the actual event vales.
            predicted (np.ndarray):
                A numpy array of shape ``(n_samples, n_horizons_timesteps, n_features)``
                with the predicted risk estimates.
            horizons (data_typing.TimeIndex):
                List of horizons time points.
            *args (Any):
                Additional positional arguments.
            **kwargs (Any):
                Additional keyword arguments.

        Returns:
            List[float]: The metric values for each horizon time point.
        """
        return super().evaluate(actual, predicted, horizons, *args, **kwargs)

    @abc.abstractmethod
    def _evaluate(  # pylint: disable=arguments-differ
        self,
        actual: metric_typing.EventArrayTimeArray,
        predicted: np.ndarray,
        horizons: data_typing.TimeIndex,
        *args: Any,
        **kwargs: Any,
    ) -> List[float]:  # pragma: no cover # noqa: D102
        ...


plugins.register_plugin_category("time_to_event", TimeToEventMetric, plugin_type="metric")
