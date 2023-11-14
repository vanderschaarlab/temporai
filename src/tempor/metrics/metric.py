"""Module containing the base class for metrics."""

# pylint: disable=unnecessary-ellipsis

import abc
from typing import Any, Generator

import rich.pretty
from typing_extensions import Literal

import tempor.core.utils
from tempor.core import plugins

MetricDirection = Literal["minimize", "maximize"]
"""The direction of the metric that represents the optimization goal (the "good" direction):
``"minimize"`` or "``maximize``".
"""


class Metric(plugins.Plugin, abc.ABC):
    """Metric abstract base class, defines the required methods."""

    @property
    @abc.abstractmethod
    def direction(self) -> MetricDirection:  # pragma: no cover
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
            Any: Evaluated metric value(s)
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
# TODO: Update the abstract methods for each case properly.


class OneOffPredictionMetric(Metric):
    """Metric abstract base class for the one-off prediction task."""

    # Overridden for type hinting.
    @abc.abstractmethod
    def _evaluate(
        self, actual: Any, predicted: Any, *args: Any, **kwargs: Any
    ) -> float:  # pragma: no cover # noqa: D102
        ...


class OneOffClassificationMetric(OneOffPredictionMetric):
    """Metric abstract base class for the one-off prediction task, classification case.

    ``predicted`` must be the probabilities in this case
    """

    # Overridden for type hinting.
    @abc.abstractmethod
    def _evaluate(
        self, actual: Any, predicted: Any, *args: Any, **kwargs: Any
    ) -> float:  # pragma: no cover # noqa: D102
        ...


plugins.register_plugin_category("prediction.one_off.classification", OneOffClassificationMetric, plugin_type="metric")


class OneOffRegressionMetric(OneOffPredictionMetric):
    """Metric abstract base class for the one-off prediction task, regression case."""

    # Overridden for type hinting.
    @abc.abstractmethod
    def _evaluate(
        self, actual: Any, predicted: Any, *args: Any, **kwargs: Any
    ) -> float:  # pragma: no cover # noqa: D102
        ...


plugins.register_plugin_category("prediction.one_off.regression", OneOffRegressionMetric, plugin_type="metric")


class TimeToEventMetric(Metric):
    """Metric abstract base class for the time-to-event (survival) task."""

    # Overridden for type hinting.
    @abc.abstractmethod
    def _evaluate(
        self, actual: Any, predicted: Any, *args: Any, **kwargs: Any
    ) -> float:  # pragma: no cover # noqa: D102
        ...


plugins.register_plugin_category("time_to_event", TimeToEventMetric, plugin_type="metric")
