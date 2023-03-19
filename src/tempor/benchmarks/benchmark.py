from typing import Any, Dict, List, Tuple

import pandas as pd
import pydantic

from tempor.data import dataset
from tempor.log import logger as log

from .evaluation import evaluate_classifier, evaluate_regressor


def print_score(mean: pd.Series, std: pd.Series) -> pd.Series:
    with pd.option_context("mode.chained_assignment", None):
        mean.loc[(mean < 1e-3) & (mean != 0)] = 1e-3
        std.loc[(std < 1e-3) & (std != 0)] = 1e-3

        mean = mean.round(3).astype(str)
        std = std.round(3).astype(str)

    return mean + " +/- " + std


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def benchmark_models(
    task_type: str,
    tests: List[Tuple[str, Any]],  # [ ( Test name, Model to evaluate (unfitted) ), ... ]
    data: dataset.Dataset,
    n_splits: int = 3,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Benchmark the performance of several algorithms.

    Args:
        task_type (str):
            The type of problem. Relevant for evaluating the downstream models with the correct metrics.
            Valid tasks are:  ``"classification"``, ``"regression"``.
        tests (List[Tuple[str, Any]]):
            Tuples of form ``(test_name: str, plugin: BasePredictor/Pipeline)``
        data (dataset.Dataset):
            The evaluation dataset to use for cross-validation.
        n_splits (int, optional):
            Number of splits used for cross-validation. Defaults to ``3``.
        random_state (int, optional):
            Random seed. Defaults to ``0``.

    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
            The benchmarking results given as ``(readable_dataframe: pd.DataFrame, results: Dict[str, pd.DataFrame]])``
            where:
            * ``readable_dataframe``: a dataframe with metric name as index and test names as columns, where the values
            are readable string representations of the evaluation metric, like: ``MEAN +/- STDDEV``.
            * ``results``: a dictionary mapping the test name to a dataframe with metric names as index and
            ``["mean", "stddev"]`` columns, where the values are the ``float`` mean and standard deviation
            for each metric.
    """

    results = {}

    if task_type == "classification":
        evaluator = evaluate_classifier
    elif task_type == "regression":
        evaluator = evaluate_regressor
    else:
        raise ValueError(f"Unsupported task type {task_type}")

    for testcase, plugin in tests:
        log.info(f"Testcase : {testcase}")

        scores = evaluator(plugin, data=data, n_splits=n_splits, random_state=random_state)

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
