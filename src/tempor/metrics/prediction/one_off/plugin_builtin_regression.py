"""Module with built-in metric plugins for the category: prediction -> one-off -> regression."""

from typing import Any, cast

import numpy as np
import sklearn.metrics

from tempor.core import plugins
from tempor.metrics import metric, metric_typing


@plugins.register_plugin(name="mse", category="prediction.one_off.regression", plugin_type="metric")
class MseOneOffRegressionMetric(metric.OneOffRegressionMetric):
    """Mean squared error regression metric"""

    @property
    def direction(self) -> metric_typing.MetricDirection:  # noqa: D102
        return "minimize"

    def _evaluate(self, actual: np.ndarray, predicted: np.ndarray, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.mean_squared_error(actual, predicted),
        )


@plugins.register_plugin(name="mae", category="prediction.one_off.regression", plugin_type="metric")
class MaeOneOffRegressionMetric(metric.OneOffRegressionMetric):
    """Mean absolute error regression metric"""

    @property
    def direction(self) -> metric_typing.MetricDirection:  # noqa: D102
        return "minimize"

    def _evaluate(self, actual: np.ndarray, predicted: np.ndarray, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.mean_absolute_error(actual, predicted),
        )


@plugins.register_plugin(name="r2", category="prediction.one_off.regression", plugin_type="metric")
class R2OneOffRegressionMetric(metric.OneOffRegressionMetric):
    """R^2 (coefficient of determination) score regression metric"""

    @property
    def direction(self) -> metric_typing.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: np.ndarray, predicted: np.ndarray, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.r2_score(actual, predicted),
        )
