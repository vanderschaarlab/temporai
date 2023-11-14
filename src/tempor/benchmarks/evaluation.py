"""Module with helpers for evaluating the performance of the methods."""

import copy
import warnings
from time import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, cast

import numpy as np
import pandas as pd
import pydantic
import scipy.stats
import sklearn.metrics
import sklearn.model_selection
from typing_extensions import Literal, get_args

from tempor.core import plugins, pydantic_utils
from tempor.data import data_typing, dataset, samples
from tempor.log import logger
from tempor.metrics.metric import OneOffClassificationMetric
from tempor.models.utils import enable_reproducibility

from . import surv_metrics, utils

if TYPE_CHECKING:  # pragma: no cover
    from tempor.methods.prediction.one_off.classification import BaseOneOffClassifier
    from tempor.methods.prediction.one_off.regression import BaseOneOffRegressor
    from tempor.methods.time_to_event import BaseTimeToEventAnalysis

# TODO: Benchmarking workflow for missing cases.

_plugin_loader = plugins.PluginLoader()
builtin_metrics_prediction_oneoff_classification = _plugin_loader.list(plugin_type="metric")["prediction"]["one_off"][
    "classification"
]

RegressionSupportedMetric = Literal[
    "mse",
    "mae",
    "r2",
]
"""Evaluation metrics supported in the regression task setting.

Possible values:
    - ``"r2"``:
        R^2 (coefficient of determination) regression score function.
    - ``"mse"``:
        Mean squared error regression loss.
    - ``"mae"``:
        Mean absolute error regression loss.
"""

TimeToEventSupportedMetric = Literal[
    "c_index",
    "brier_score",
]
"""Evaluation metrics supported in the time-to-event (survival) task setting.

Possible values:
    - ``"c_index"``:
        Concordance index based on inverse probability of censoring weights.
    - ``"brier_score"``:
        The time-dependent Brier score.
"""

OutputMetric = Literal[
    "min",
    "max",
    "mean",
    "stddev",
    "median",
    "iqr",
    "rounds",
    "errors",
    "durations",
]
"""The metric evaluation output statistics / other information about the evaluation cross-validation runs.

Possible values:
    - ``"min"``:
        The mix score of the metric
    - ``"max"``:
        The max score of the metric
    - ``"mean"``:
        The mean score of the metric
    - ``"stddev"``:
        The stddev score of the metric
    - ``"median"``:
        The median score of the metric
    - ``"iqr"``:
        The interquartile range of the metric
    - ``"rounds"``:
        Number of folds
    - ``"errors"``:
        Number of errors encountered
    - ``"durations"``:
        Average duration for the fold evaluation.
"""

regression_supported_metrics = get_args(RegressionSupportedMetric)
"""A tuple of all possible values of :obj:`~tempor.benchmarks.evaluation.RegressionSupportedMetric`."""

time_to_event_supported_metrics = get_args(TimeToEventSupportedMetric)
"""A tuple of all possible values of :obj:`~tempor.benchmarks.evaluation.TimeToEventSupportedMetric`."""

output_metrics = get_args(OutputMetric)
"""A tuple of all possible values of :obj:`~tempor.benchmarks.evaluation.OutputMetric`."""


class _InternalScores(pydantic.BaseModel):
    metrics: Dict[str, np.ndarray] = {}  # np.ndarray expected to be 1D, contain floats.
    errors: List[int] = []
    durations: List[float] = []
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


@pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
def _postprocess_results(results: _InternalScores) -> pd.DataFrame:
    output = pd.DataFrame([], columns=output_metrics)

    for metric in results.metrics:
        values = results.metrics[metric]
        errors = np.sum(results.errors)
        durations = utils.print_score(utils.generate_score(np.asarray(results.durations)))

        score_min = np.min(values)
        score_max = np.max(values)
        score_mean = np.mean(values)
        score_median = np.median(values)
        score_stddev = np.std(values)
        score_iqr = scipy.stats.iqr(values)
        score_rounds = len(values)

        output = pd.concat(
            [
                output,
                pd.DataFrame(
                    [
                        [
                            score_min,
                            score_max,
                            score_mean,
                            score_stddev,
                            score_median,
                            score_iqr,
                            score_rounds,
                            errors,
                            durations,
                        ]
                    ],
                    columns=output_metrics,
                    index=[metric],
                ),
            ],
        )

    return output


@pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
def evaluate_prediction_oneoff_classifier(  # pylint: disable=unused-argument
    estimator: Any,
    data: dataset.PredictiveDataset,
    n_splits: int = 3,
    random_state: int = 0,
    raise_exceptions: bool = False,
    silence_warnings: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Helper for evaluating classifiers.

    Args:
        estimator (Any):
            Baseline model to evaluate - must be unfitted.
        data (dataset.PredictiveDataset):
            The dataset.
        n_splits (int, optional):
            Cross-validation folds. Defaults to ``3``.
        random_state (int, optional):
            Random state. Defaults to ``0``.
        raise_exceptions (bool, optional):
            Whether to raise exceptions during evaluation. If `False`, the exceptions will be swallowed and the
            evaluation will continue - exception count will be reported in the `"errors"` column of the resultant
            dataframe. Defaults to `False`.
        silence_warnings (bool, optional):
            Whether to silence warnings raised. Defaults to `False`.
        **kwargs (Any):
            Currently unused.

    Returns:
        pd.DataFrame:
            DataFrame containing the results.

            The columns of the dataframe contain details about the cross-validation repeats: one column for each
            :obj:`~tempor.benchmarks.evaluation.OutputMetric`.

            The index of the dataframe contains all the metrics evaluated: all metric plugins registered:

            >>> import doctest; doctest.ELLIPSIS_MARKER = "[...]"  # Doctest config, ignore.
            >>> from tempor import plugin_loader
            >>> plugin_loader.list(plugin_type="metric")["prediction"]["one_off"]["classification"]
            [...]
    """

    # For the sake of import modularity, do not use the global plugin loader here, but create own:
    _plugin_loader = plugins.PluginLoader()
    metric_plugin_category = "prediction.one_off.classification"
    classifier_supported_metrics = _plugin_loader.list(plugin_type="metric")["prediction"]["one_off"]["classification"]

    with warnings.catch_warnings():
        if silence_warnings:
            warnings.simplefilter("ignore")

        if n_splits < 2 or not isinstance(n_splits, int):
            raise ValueError("n_splits must be an integer >= 2")
        estimator_ = cast("BaseOneOffClassifier", estimator)
        enable_reproducibility(random_state)

        results = _InternalScores()
        for metric_name in classifier_supported_metrics:
            results.metrics[metric_name] = np.zeros(n_splits)

        splitter = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        if data.predictive.targets is None:
            raise ValueError("The dataset for evaluation needs to contain targets but did not")
        labels = data.predictive.targets.numpy().squeeze()
        if len(labels.shape) > 1:
            raise ValueError("Classifier evaluation expects 1D output")

        indx = 0
        for train_data, test_data in data.split(splitter=splitter, y=labels):
            model = copy.deepcopy(estimator_)
            start = time()

            try:
                model.fit(train_data)

                if TYPE_CHECKING:  # pragma: no cover
                    assert test_data.predictive.targets is not None  # nosec B101
                test_labels = test_data.predictive.targets.numpy()
                preds = model.predict_proba(test_data).numpy()

                for metric_name in classifier_supported_metrics:
                    metric = cast(
                        OneOffClassificationMetric,
                        _plugin_loader.get(f"{metric_plugin_category}.{metric_name}", plugin_type="metric"),
                    )
                    results.metrics[metric_name][indx] = metric.evaluate(test_labels, preds)

                results.errors.append(0)

            except BaseException as e:  # pylint: disable=broad-except
                logger.error(f"Evaluation failed: {e}")
                results.errors.append(1)
                if raise_exceptions:
                    raise

            results.durations.append(time() - start)
            indx += 1

    return _postprocess_results(results)


@pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
def evaluate_prediction_oneoff_regressor(  # pylint: disable=unused-argument
    estimator: Any,
    data: dataset.PredictiveDataset,
    n_splits: int = 3,
    random_state: int = 0,
    raise_exceptions: bool = False,
    silence_warnings: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Helper for evaluating regression tasks.

    Args:
        estimator (Any):
            Baseline model to evaluate - must be unfitted.
        data (dataset.PredictiveDataset):
            The dataset.
        n_splits (int, optional):
            Cross-validation folds. Defaults to ``3``.
        random_state (int, optional):
            Random state. Defaults to ``0``.
        raise_exceptions (bool, optional):
            Whether to raise exceptions during evaluation. If `False`, the exceptions will be swallowed and the
            evaluation will continue - exception count will be reported in the `"errors"` column of the resultant
            dataframe. Defaults to `False`.
        silence_warnings (bool, optional):
            Whether to silence warnings raised. Defaults to `False`.
        **kwargs (Any):
            Currently unused.

    Returns:
        pd.DataFrame:
            DataFrame containing the results.

            The columns of the dataframe contain details about the cross-validation repeats: one column for each
            :obj:`~tempor.benchmarks.evaluation.OutputMetric`.

            The index of the dataframe contains all the metrics evaluated: all of
            :obj:`~tempor.benchmarks.evaluation.RegressionSupportedMetric`.
    """

    with warnings.catch_warnings():
        if silence_warnings:
            warnings.simplefilter("ignore")

        if n_splits < 2 or not isinstance(n_splits, int):
            raise ValueError("n_splits must be an integer >= 2")
        estimator_ = cast("BaseOneOffRegressor", estimator)
        enable_reproducibility(random_state)
        metrics = regression_supported_metrics

        results = _InternalScores()
        for metric in metrics:
            results.metrics[metric] = np.zeros(n_splits)

        splitter = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        indx = 0
        for train_data, test_data in data.split(splitter=splitter):
            model = copy.deepcopy(estimator_)
            start = time()
            try:
                model.fit(train_data)

                if TYPE_CHECKING:  # pragma: no cover
                    assert test_data.predictive.targets is not None  # nosec B101
                targets = test_data.predictive.targets.numpy().squeeze()
                preds = model.predict(test_data).numpy().squeeze()

                results.metrics["mse"][indx] = sklearn.metrics.mean_squared_error(targets, preds)
                results.metrics["mae"][indx] = sklearn.metrics.mean_absolute_error(targets, preds)
                results.metrics["r2"][indx] = sklearn.metrics.r2_score(targets, preds)
                results.errors.append(0)
            except BaseException as e:  # pylint: disable=broad-except
                logger.error(f"Regression evaluation failed: {e}")
                results.errors.append(1)
                if raise_exceptions:
                    raise

            results.durations.append(time() - start)
            indx += 1

    return _postprocess_results(results)


TimeToEventMetricCallable = Callable[[np.ndarray, np.ndarray, np.ndarray, List[float]], List[float]]
"""Standardized function for time-to-event metric.

Inputs are:
    * ``training_array_struct`` (np.ndarray)
    * ``testing_array_struct`` (np.ndarray)
    * ``predictions`` (np.ndarray)
    * ``horizons`` (List[float])

Output is:
    A list with the metric values for each horizon.
"""


def compute_c_index(
    training_array_struct: np.ndarray, testing_array_struct: np.ndarray, predictions: np.ndarray, horizons: List[float]
) -> List[float]:
    """Compute the IPCW concordance index.

    Args:
        training_array_struct (np.ndarray): Training data as a structured array.
        testing_array_struct (np.ndarray): Testing data as a structured array.
        predictions (np.ndarray): Predictions.
        horizons (List[float]): Evaluation horizons.

    Returns:
        List[float]: List of metric values for each horizon.
    """
    metrics: List[float] = []
    for horizon_idx, horizon_time in enumerate(horizons):
        predictions_at_horizon_time = predictions[:, horizon_idx, :].reshape((-1,))
        out = surv_metrics.concordance_index_ipcw(
            training_array_struct, testing_array_struct, predictions_at_horizon_time, float(horizon_time)
        )
        metrics.append(out[0])
    return metrics


def compute_brier_score(
    training_array_struct: np.ndarray, testing_array_struct: np.ndarray, predictions: np.ndarray, horizons: List[float]
) -> List[float]:
    """Compute the time-dependent Brier score.

    Args:
        training_array_struct (np.ndarray): Training data as a structured array.
        testing_array_struct (np.ndarray): Testing data as a structured array.
        predictions (np.ndarray): Predictions.
        horizons (List[float]): Evaluation horizons.

    Returns:
        List[float]: List of metric values for each horizon.
    """
    predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]))
    times, scores = surv_metrics.brier_score(  # pylint: disable=unused-variable
        training_array_struct, testing_array_struct, predictions, horizons
    )
    return scores.tolist()


def _compute_time_to_event_metric(
    metric_func: TimeToEventMetricCallable,
    train_data: dataset.TimeToEventAnalysisDataset,
    test_data: dataset.TimeToEventAnalysisDataset,
    horizons: data_typing.TimeIndex,
    predictions: samples.TimeSeriesSamples,
) -> float:
    if not predictions.num_timesteps_equal():
        raise ValueError(
            f"Expected time to event prediction values for horizons {horizons} all to have equal number of time steps "
            f"({len(horizons)} but different lengths found {predictions.num_timesteps()}"
        )
    if train_data.predictive.targets is None or test_data.predictive.targets is None:
        raise ValueError("Expected data to have targets but did not")
    for data, name in zip(
        (predictions, train_data.predictive.targets, test_data.predictive.targets),
        ("predictions", "training data targets", "testing data targets"),
    ):
        if data.num_features > 1:
            raise ValueError(
                "Currently time to event evaluation only supports one event "
                f"but more than one event features found in {name}"
            )
    try:
        float(horizons[0])  # pyright: ignore
    except (TypeError, ValueError) as e:
        raise ValueError("Currently only int or float time horizons supported.") from e
    horizons = cast(List[float], horizons)

    predictions_array = predictions.numpy()
    t_train, y_train = (df.to_numpy().reshape((-1,)) for df in train_data.predictive.targets.split_as_two_dataframes())
    y_train_struct = surv_metrics.create_structured_array(y_train, t_train)
    t_test, y_test = (df.to_numpy().reshape((-1,)) for df in test_data.predictive.targets.split_as_two_dataframes())
    y_test_struct = surv_metrics.create_structured_array(y_test, t_test)

    metrics: List[float] = metric_func(y_train_struct, y_test_struct, predictions_array, horizons)
    avg_metric = float(np.asarray(metrics).mean())

    return avg_metric


@pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
def evaluate_time_to_event(  # pylint: disable=unused-argument
    estimator: Any,
    data: dataset.TimeToEventAnalysisDataset,
    horizons: data_typing.TimeIndex,
    n_splits: int = 3,
    random_state: int = 0,
    raise_exceptions: bool = False,
    silence_warnings: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Helper for evaluating time-to-event tasks.

    Args:
        estimator (Any):
            Baseline model to evaluate - must be unfitted
        data (dataset.TimeToEventAnalysisDataset):
             The dataset.
        horizons (data_typing.TimeIndex):
            Time horizons for making predictions at.
        n_splits (int, optional):
            Cross-validation folds. Defaults to ``3``.
        random_state (int, optional):
            Random state. Defaults to ``0``.
        raise_exceptions (bool, optional):
            Whether to raise exceptions during evaluation. If `False`, the exceptions will be swallowed and the
            evaluation will continue - exception count will be reported in the `"errors"` column of the resultant
            dataframe. Defaults to `False`.
        silence_warnings (bool, optional):
            Whether to silence warnings raised. Defaults to `False`.
        **kwargs (Any):
            Currently unused.

    Returns:
        pd.DataFrame:
            DataFrame containing the results.

            The columns of the dataframe contain details about the cross-validation repeats: one column for each
            :obj:`~tempor.benchmarks.evaluation.OutputMetric`.

            The index of the dataframe contains all the metrics evaluated: all of
            :obj:`~tempor.benchmarks.evaluation.TimeToEventSupportedMetric`.
    """
    with warnings.catch_warnings():
        if silence_warnings:
            warnings.simplefilter("ignore")
            # NOTE: xbgse is somehow able to circumvent warnings silencing, so will still raise warnings.

        if n_splits < 2 or not isinstance(n_splits, int):
            raise ValueError("n_splits must be an integer >= 2")
        estimator_ = cast("BaseTimeToEventAnalysis", estimator)
        enable_reproducibility(random_state)
        metrics = time_to_event_supported_metrics
        metrics_map = {
            "c_index": compute_c_index,
            "brier_score": compute_brier_score,
        }

        results = _InternalScores()
        for metric in metrics:
            results.metrics[metric] = np.zeros(n_splits)

        splitter = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        indx = 0
        for train_data, test_data in data.split(splitter=splitter):
            model = copy.deepcopy(estimator_)
            start = time()
            try:
                model.fit(train_data)

                # targets = test_data.predictive.targets.numpy().squeeze()
                preds = model.predict(test_data, horizons=horizons)

                for metric_name in time_to_event_supported_metrics:
                    metric_func = metrics_map[metric_name]
                    results.metrics[metric_name][indx] = _compute_time_to_event_metric(
                        metric_func,
                        train_data=train_data,
                        test_data=test_data,
                        horizons=horizons,
                        predictions=preds,
                    )

                results.errors.append(0)

            except BaseException as e:  # pylint: disable=broad-except
                logger.error(f"Regression evaluation failed: {e}")
                results.errors.append(1)
                if raise_exceptions:
                    raise

            results.durations.append(time() - start)
            indx += 1

    return _postprocess_results(results)
