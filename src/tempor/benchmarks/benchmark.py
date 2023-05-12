from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pydantic
from typing_extensions import Literal

from tempor.data import data_typing, dataset
from tempor.log import logger as log

from . import evaluation

TaskType = Literal[
    "prediction.one_off.classification",
    "prediction.one_off.regression",
    "prediction.temporal.classification",
    "prediction.temporal.regression",
    "time_to_event",
    "treatments.one_off.classification",
    "treatments.one_off.regression",
    "treatments.temporal.classification",
    "treatments.temporal.regression",
]


def print_score(mean: pd.Series, std: pd.Series) -> pd.Series:
    with pd.option_context("mode.chained_assignment", None):
        mean.loc[(mean < 1e-3) & (mean != 0)] = 1e-3
        std.loc[(std < 1e-3) & (std != 0)] = 1e-3

        mean = mean.round(3).astype(str)
        std = std.round(3).astype(str)

    return mean + " +/- " + std


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def benchmark_models(
    task_type: TaskType,
    tests: List[Tuple[str, Any]],  # [ ( Test name, Model to evaluate (unfitted) ), ... ]
    data: dataset.PredictiveDataset,
    n_splits: int = 3,
    random_state: int = 0,
    horizons: Optional[data_typing.TimeIndex] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Benchmark the performance of several algorithms.

    Args:
        task_type (TaskType):
            The type of problem. Relevant for evaluating the downstream models with the correct metrics.
            The options are any of `TaskType`.
        tests (List[Tuple[str, Any]]):
            Tuples of form ``(test_name: str, plugin: BasePredictor/Pipeline)``
        data (dataset.Dataset):
            The evaluation dataset to use for cross-validation.
        n_splits (int, optional):
            Number of splits used for cross-validation. Defaults to ``3``.
        random_state (int, optional):
            Random seed. Defaults to ``0``.
        horizons (data_typing.TimeIndex, optional):
            Time horizons for making predictions, if applicable to the task.

    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
            The benchmarking results given as ``(readable_dataframe: pd.DataFrame, results: Dict[str, pd.DataFrame]])``\
                where:
                * ``readable_dataframe``: a dataframe with metric name as index and test names as columns, where the\
                    values are readable string representations of the evaluation metric, like: ``MEAN +/- STDDEV``.
                * ``results``: a dictionary mapping the test name to a dataframe with metric names as index and\
                    ``["mean", "stddev"]`` columns, where the values are the ``float`` mean and standard deviation\
                    for each metric.
    """

    results = {}

    if task_type == "prediction.one_off.classification":
        evaluator: Callable = evaluation.evaluate_prediction_oneoff_classifier
    elif task_type == "prediction.one_off.regression":
        evaluator = evaluation.evaluate_prediction_oneoff_regressor
    elif task_type == "time_to_event":
        evaluator = evaluation.evaluate_time_to_event
    elif task_type == "prediction.temporal.classification":
        # TODO: This case needs to be handled in `tempor.benchmarks.evaluation`.
        raise NotImplementedError
    elif task_type == "prediction.temporal.regression":
        # TODO: This case needs to be handled in `tempor.benchmarks.evaluation`.
        raise NotImplementedError
    elif task_type == "treatments.one_off.classification":
        # TODO: This case needs to be handled in `tempor.benchmarks.evaluation`.
        raise NotImplementedError
    elif task_type == "treatments.one_off.regression":
        # TODO: This case needs to be handled in `tempor.benchmarks.evaluation`.
        raise NotImplementedError
    elif task_type == "treatments.temporal.classification":
        # TODO: This case needs to be handled in `tempor.benchmarks.evaluation`.
        raise NotImplementedError
    elif task_type == "treatments.temporal.regression":
        # TODO: This case needs to be handled in `tempor.benchmarks.evaluation`.
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    for testcase, plugin in tests:
        log.info(f"Test case: {testcase}")

        scores = evaluator(
            plugin,
            data=data,  # type: ignore
            n_splits=n_splits,
            random_state=random_state,
            horizons=horizons,  # type: ignore
        )

        mean_score = scores["mean"].to_dict()
        stddev_score = scores["stddev"].to_dict()

        local_scores = {}
        for key in mean_score:
            local_scores[key] = {
                "mean": mean_score[key],
                "stddev": stddev_score[key],
            }
        results[testcase] = pd.DataFrame(local_scores).T

    means = []
    for testcase in results:
        mean = results[testcase]["mean"]
        stddev = results[testcase]["stddev"]
        means.append(print_score(mean, stddev))

    aggr = pd.concat(means, axis=1)
    aggr = aggr.set_axis(results.keys(), axis=1)

    return aggr, results
