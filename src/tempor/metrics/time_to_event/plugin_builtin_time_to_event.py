"""Module with built-in metric plugins for the time-to-event (survival) analysis task."""
from typing import Any, Callable, List

import numpy as np

from tempor.core import plugins
from tempor.data import data_typing
from tempor.metrics import metric, metric_typing

from . import _metric_impl


def _prep_surv_metric_data(
    metric_func: Callable,
    y_test: np.ndarray,
    t_test: np.ndarray,
    predictions_array: np.ndarray,
    horizons: List[float],
    y_train: np.ndarray,
    t_train: np.ndarray,
) -> List[float]:
    """Prepare the input arrays for the survival metrics and return the metric per horizon.

    Args:
        metric_func (Callable): Metric function to be used.
        y_test (np.ndarray): Event values, test set.
        t_test (np.ndarray): Event time points, test set.
        predictions_array (np.ndarray): Predictions array of shape ``(n_samples, n_horizons_timesteps, n_features)``
        horizons (List[float]): List of horizons time points.
        y_train (np.ndarray): Event values, training set.
        t_train (np.ndarray): Event time points, training set.

    Returns:
        List[float]: Metric values for each horizon time point.
    """
    y_train_struct = _metric_impl.create_structured_array(y_train, t_train)
    y_test_struct = _metric_impl.create_structured_array(y_test, t_test)

    metric_per_horizon: List[float] = metric_func(y_train_struct, y_test_struct, predictions_array, horizons)

    return metric_per_horizon


# TODO: support non-float horizons.


@plugins.register_plugin(name="c_index", category="time_to_event", plugin_type="metric")
class CIndexTimeToEventMetric(metric.TimeToEventMetric):
    """IPCW concordance index metric for time-to-event (survival) analysis tasks."""

    @property
    def direction(self) -> metric_typing.MetricDirection:  # noqa: D102
        return "maximize"

    # Override to update the extra argument `actual_train`.
    def evaluate(  # type: ignore [override] # pylint: disable=arguments-differ
        self,
        actual: metric_typing.EventArrayTimeArray,
        predicted: np.ndarray,
        horizons: data_typing.TimeIndex,
        actual_train: metric_typing.EventArrayTimeArray,
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
            actual_train (metric_typing.EventArrayTimeArray):
                A tuple of two numpy arrays: the event values array and the event times array,
                for the actual event vales - in the training set.
            *args (Any):
                Additional positional arguments.
            **kwargs (Any):
                Additional keyword arguments.

        Returns:
            List[float]: The metric values for each horizon time point.
        """
        return super().evaluate(actual, predicted, horizons, actual_train, *args, **kwargs)

    def _evaluate(  # type: ignore [override] # pylint: disable=arguments-differ
        self,
        actual: metric_typing.EventArrayTimeArray,
        predicted: np.ndarray,
        horizons: data_typing.TimeIndex,
        actual_train: metric_typing.EventArrayTimeArray,
        *args: Any,
        **kwargs: Any,
    ) -> List[float]:
        y_test, t_test = actual
        y_train, t_train = actual_train

        metric_per_horizon = _prep_surv_metric_data(
            _metric_impl.compute_c_index,
            y_test,
            t_test,
            predicted,
            horizons,  # type: ignore
            y_train,
            t_train,
        )
        return metric_per_horizon


@plugins.register_plugin(name="brier_score", category="time_to_event", plugin_type="metric")
class BrierScoreTimeToEventMetric(metric.TimeToEventMetric):
    """Time-dependent Brier score metric for time-to-event (survival) analysis tasks."""

    @property
    def direction(self) -> metric_typing.MetricDirection:  # noqa: D102
        return "minimize"

    # Override to update the extra argument `actual_train`.
    def evaluate(  # type: ignore [override] # pylint: disable=arguments-differ
        self,
        actual: metric_typing.EventArrayTimeArray,
        predicted: np.ndarray,
        horizons: data_typing.TimeIndex,
        actual_train: metric_typing.EventArrayTimeArray,
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
            actual_train (metric_typing.EventArrayTimeArray):
                A tuple of two numpy arrays: the event values array and the event times array,
                for the actual event vales - in the training set.
            *args (Any):
                Additional positional arguments.
            **kwargs (Any):
                Additional keyword arguments.

        Returns:
            List[float]: The metric values for each horizon time point.
        """
        return super().evaluate(actual, predicted, horizons, actual_train, *args, **kwargs)

    def _evaluate(  # type: ignore [override] # pylint: disable=arguments-differ
        self,
        actual: metric_typing.EventArrayTimeArray,
        predicted: np.ndarray,
        horizons: data_typing.TimeIndex,
        actual_train: metric_typing.EventArrayTimeArray,
        *args: Any,
        **kwargs: Any,
    ) -> List[float]:
        y_test, t_test = actual
        y_train, t_train = actual_train

        metric_per_horizon = _prep_surv_metric_data(
            _metric_impl.compute_brier_score,
            y_test,
            t_test,
            predicted,
            horizons,  # type: ignore
            y_train,
            t_train,
        )
        return metric_per_horizon
