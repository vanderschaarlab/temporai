import sys
from typing import Callable

import pytest

from tempor import plugin_loader
from tempor.benchmarks import (
    benchmark_models,
    builtin_metrics_prediction_oneoff_classification,
    builtin_metrics_prediction_oneoff_regression,
    time_to_event_supported_metrics,
    visualize_benchmark,
)
from tempor.methods.pipeline import pipeline

N_ITER = 5

TEST_ON_DATASETS_CLASSIFIER = ["sine_data_small"]
TEST_ON_DATASETS_REGRESSOR = ["google_stocks_data_small"]
TEST_ON_DATASETS_TIME_TO_EVENT = ["pbc_data_small"]

PREDICTOR_CLASSIFICATION = "prediction.one_off.classification.nn_classifier"
PREDICTOR_REGRESSION = "prediction.one_off.regression.nn_regressor"
PREDICTOR_TIME_TO_EVENT = "time_to_event.dynamic_deephit"


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*All-NaN.*:RuntimeWarning")  # Expected matplotlib warnings.
@pytest.mark.filterwarnings("ignore:.*Matplotlib.*:UserWarning")
@pytest.mark.parametrize("data", TEST_ON_DATASETS_CLASSIFIER)
def test_classifier_benchmark(data: str, request: pytest.FixtureRequest) -> None:
    testcases = [
        (
            "pipeline1",
            pipeline(
                [
                    "preprocessing.imputation.static.static_tabular_imputer",
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
        task_type="prediction.one_off.classification",
        tests=testcases,
        data=dataset,
        n_splits=2,
        random_state=0,
    )

    for testcase, _ in testcases:
        assert testcase in aggr_score.columns
        assert testcase in per_test_score

    for metric in builtin_metrics_prediction_oneoff_classification:
        assert metric in aggr_score.index

        for testcase, _ in testcases:
            assert metric in per_test_score[testcase].index

    # Check also visualize_benchmarks executes without error.
    if not sys.platform.startswith(("win", "darwin")):
        # TODO: There appears to be a problem running this on GH runners for Windows and MaxOS, hence disabled.
        # Investigate and resolve.
        visualize_benchmark(per_test_score, plot_block=False)


@pytest.mark.parametrize("data", TEST_ON_DATASETS_REGRESSOR)
def test_regressor_benchmark(data: str, request: pytest.FixtureRequest) -> None:
    testcases = [
        (
            "pipeline1",
            pipeline(
                [
                    "preprocessing.imputation.static.static_tabular_imputer",
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
        task_type="prediction.one_off.regression",
        tests=testcases,
        data=dataset,
        n_splits=2,
        random_state=0,
    )

    for testcase, _ in testcases:
        assert testcase in aggr_score.columns
        assert testcase in per_test_score

    for metric in builtin_metrics_prediction_oneoff_regression:
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
            pipeline(
                [
                    "preprocessing.imputation.static.static_tabular_imputer",
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
