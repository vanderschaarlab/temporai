from typing import Any, Callable

import pytest

from tempor.benchmarks import (
    classifier_supported_metrics,
    evaluate_classifier,
    evaluate_regressor,
    evaluate_time_to_event,
    output_metrics,
    regression_supported_metrics,
    time_to_event_supported_metrics,
)
from tempor.plugins import plugin_loader
from tempor.plugins.pipeline import pipeline

N_ITER = 5

TEST_ON_DATASETS_CLASSIFIER = ["sine_data_small"]
TEST_ON_DATASETS_REGRESSOR = ["google_stocks_data_small"]
TEST_ON_DATASETS_TIME_TO_EVENT = ["pbc_data_small"]

PREDICTOR_CLASSIFICATION = "prediction.one_off.classification.nn_classifier"
PREDICTOR_REGRESSION = "prediction.one_off.regression.nn_regressor"
PREDICTOR_TIME_TO_EVENT = "time_to_event.dynamic_deephit"


@pytest.mark.parametrize("data", TEST_ON_DATASETS_CLASSIFIER)
@pytest.mark.parametrize(
    "model_template",
    [
        pipeline(
            [
                "preprocessing.imputation.temporal.ffill",
                PREDICTOR_CLASSIFICATION,
            ]
        )({"nn_classifier": {"n_iter": N_ITER}}),
        pipeline(
            [
                PREDICTOR_CLASSIFICATION,
            ]
        )({"nn_classifier": {"n_iter": N_ITER}}),
        plugin_loader.get(PREDICTOR_CLASSIFICATION, n_iter=N_ITER),
    ],
)
@pytest.mark.parametrize("n_splits", [2])
def test_classifier_evaluation(model_template: Any, n_splits: int, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    scores = evaluate_classifier(model_template, dataset, n_splits=n_splits, seed=0)

    for out_metric in output_metrics:
        assert out_metric in scores

    for metric in classifier_supported_metrics:
        assert metric in scores.index


@pytest.mark.parametrize("data", TEST_ON_DATASETS_CLASSIFIER)
@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_classifier_evaluation_fail(n_splits: int, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    with pytest.raises(ValueError):
        evaluate_classifier(
            plugin_loader.get(PREDICTOR_CLASSIFICATION, n_iter=N_ITER), dataset, n_splits=n_splits, seed=0
        )


@pytest.mark.parametrize("data", TEST_ON_DATASETS_REGRESSOR)
@pytest.mark.parametrize(
    "model_template",
    [
        pipeline(
            [
                "preprocessing.imputation.temporal.ffill",
                PREDICTOR_REGRESSION,
            ]
        )({"nn_regressor": {"n_iter": N_ITER}}),
        pipeline(
            [
                PREDICTOR_REGRESSION,
            ]
        )({"nn_regressor": {"n_iter": N_ITER}}),
        plugin_loader.get(PREDICTOR_REGRESSION, n_iter=N_ITER),
    ],
)
@pytest.mark.parametrize("n_splits", [2])
def test_regressor_evaluation(model_template: Any, n_splits: int, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    scores = evaluate_regressor(model_template, dataset, n_splits=n_splits, seed=0)

    for out_metric in output_metrics:
        assert out_metric in scores

    for metric in regression_supported_metrics:
        assert metric in scores.index


@pytest.mark.parametrize("data", TEST_ON_DATASETS_REGRESSOR)
@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_regressor_evaluation_fail(n_splits: int, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    with pytest.raises(ValueError):
        evaluate_regressor(plugin_loader.get(PREDICTOR_REGRESSION, n_iter=N_ITER), dataset, n_splits=n_splits, seed=0)


@pytest.mark.parametrize("data", TEST_ON_DATASETS_TIME_TO_EVENT)
@pytest.mark.parametrize(
    "model_template",
    [
        pipeline(
            [
                "preprocessing.imputation.temporal.ffill",
                PREDICTOR_TIME_TO_EVENT,
            ]
        )({"dynamic_deephit": {"n_iter": N_ITER}}),
        pipeline(
            [
                PREDICTOR_TIME_TO_EVENT,
            ]
        )({"dynamic_deephit": {"n_iter": N_ITER}}),
        plugin_loader.get(PREDICTOR_TIME_TO_EVENT, n_iter=N_ITER),
    ],
)
@pytest.mark.parametrize("n_splits", [2])
def test_time_to_event_evaluation(
    model_template: Any, n_splits: int, get_event0_time_percentiles: Callable, data: str, request: pytest.FixtureRequest
) -> None:
    dataset = request.getfixturevalue(data)

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    scores = evaluate_time_to_event(model_template, dataset, n_splits=n_splits, seed=0, horizons=horizons)

    for out_metric in output_metrics:
        assert out_metric in scores

    for metric in time_to_event_supported_metrics:
        assert metric in scores.index


@pytest.mark.parametrize("data", TEST_ON_DATASETS_TIME_TO_EVENT)
@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_time_to_event_evaluation_fail(
    n_splits: int, get_event0_time_percentiles: Callable, data: str, request: pytest.FixtureRequest
) -> None:
    dataset = request.getfixturevalue(data)

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    with pytest.raises(ValueError):
        evaluate_time_to_event(
            plugin_loader.get("time_to_event.nn_regressor", n_iter=N_ITER),
            dataset,
            n_splits=n_splits,
            seed=0,
            horizons=horizons,
        )
