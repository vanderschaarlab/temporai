# stdlib
from typing import Any, Dict, List, Tuple

# third party
import pandas as pd
from pydantic import validate_arguments

# synthcity absolute
from tempor.data import dataset
from tempor.log import logger as log

from .evaluation import evaluate_classifier, evaluate_regressor


def print_score(mean: pd.Series, std: pd.Series) -> pd.Series:
    pd.options.mode.chained_assignment = None

    mean.loc[(mean < 1e-3) & (mean != 0)] = 1e-3
    std.loc[(std < 1e-3) & (std != 0)] = 1e-3

    mean = mean.round(3).astype(str)
    std = std.round(3).astype(str)

    return mean + " +/- " + std


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def benchmark_models(
    task_type: str,
    tests: List[Tuple[str, Any]],  # test name, model to evaluate(unfitted)
    data: dataset.Dataset,
    n_splits: int = 3,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Benchmark the performance of several algorithms.

    Args:
        task_type: str
            The type of problem. Relevant for evaluating the downstream models with the correct metrics. Valid tasks are:  "classification", "regression".
        tests: List[Tuple[str, Any model]]
            Tuples of form (testname: str, plugin: BasePredictor/Pipeline)
        data: Dataset
            The evaluation dataset to use for cross-validation.
        n_splits: int
            Number of splits used for cross-validation.
        random_state: int
            Random seed.
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
