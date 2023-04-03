from typing import Callable

import pytest

from tempor.benchmarks import (
    benchmark_models,
    classifier_supported_metrics,
    regression_supported_metrics,
    time_to_event_supported_metrics,
)
from tempor.plugins import plugin_loader
from tempor.plugins.pipeline import Pipeline

N_ITER = 5

TEST_ON_DATASETS_CLASSIFIER = ["sine_data_small"]
TEST_ON_DATASETS_REGRESSOR = ["google_stocks_data_small"]
TEST_ON_DATASETS_TIME_TO_EVENT = ["pbc_data_small"]

PREDICTOR_CLASSIFICATION = "prediction.one_off.classification.nn_classifier"
PREDICTOR_REGRESSION = "prediction.one_off.regression.nn_regressor"
PREDICTOR_TIME_TO_EVENT = "time_to_event.dynamic_deephit"


@pytest.mark.parametrize("data", TEST_ON_DATASETS_CLASSIFIER)
def test_classifier_benchmark(data: str, request: pytest.FixtureRequest) -> None:
    testcases = [
        (
            "pipeline1",
            Pipeline(
                [
                    "preprocessing.imputation.temporal.ffill",
                    PREDICTOR_CLASSIFICATION,
                ]
            )({"nn_classifier": {"n_iter": N_ITER}}),
        ),
        (
            "plugin1",
            plugin_loader.get(PREDICTOR_CLASSIFICATION, n_iter=N_ITER),
        ),
    ]
    dataset = request.getfixturevalue(data)

    aggr_score, per_test_score = benchmark_models(
        task_type="classification",
        tests=testcases,
        data=dataset,
        n_splits=2,
        random_state=0,
    )

    for testcase, _ in testcases:
        assert testcase in aggr_score.columns
        assert testcase in per_test_score

    for metric in classifier_supported_metrics:
        assert metric in aggr_score.index

        for testcase, _ in testcases:
            assert metric in per_test_score[testcase].index


@pytest.mark.parametrize("data", TEST_ON_DATASETS_REGRESSOR)
def test_regressor_benchmark(data: str, request: pytest.FixtureRequest) -> None:
    testcases = [
        (
            "pipeline1",
            Pipeline(
                [
                    "preprocessing.imputation.temporal.ffill",
                    PREDICTOR_REGRESSION,
                ]
            )({"nn_regressor": {"n_iter": N_ITER}}),
        ),
        (
            "plugin1",
            plugin_loader.get(PREDICTOR_REGRESSION, n_iter=N_ITER),
        ),
    ]
    dataset = request.getfixturevalue(data)

    aggr_score, per_test_score = benchmark_models(
        task_type="regression",
        tests=testcases,
        data=dataset,
        n_splits=2,
        random_state=0,
    )

    for testcase, _ in testcases:
        assert testcase in aggr_score.columns
        assert testcase in per_test_score

    for metric in regression_supported_metrics:
        assert metric in aggr_score.index

        for testcase, _ in testcases:
            assert metric in per_test_score[testcase].index


@pytest.mark.parametrize("data", TEST_ON_DATASETS_TIME_TO_EVENT)
def test_time_to_event_benchmark(
    get_event0_time_percentiles: Callable, data: str, request: pytest.FixtureRequest
) -> None:
    testcases = [
        (
            "pipeline1",
            Pipeline(
                [
                    "preprocessing.imputation.temporal.ffill",
                    PREDICTOR_TIME_TO_EVENT,
                ]
            )({"dynamic_deephit": {"n_iter": N_ITER}}),
        ),
        (
            "plugin1",
            plugin_loader.get(PREDICTOR_TIME_TO_EVENT, n_iter=N_ITER),
        ),
    ]
    dataset = request.getfixturevalue(data)

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    aggr_score, per_test_score = benchmark_models(
        task_type="time_to_event",
        tests=testcases,
        data=dataset,
        n_splits=2,
        random_state=0,
        horizons=horizons,
    )

    for testcase, _ in testcases:
        assert testcase in aggr_score.columns
        assert testcase in per_test_score

    for metric in time_to_event_supported_metrics:
        assert metric in aggr_score.index

        for testcase, _ in testcases:
            assert metric in per_test_score[testcase].index
