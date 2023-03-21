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
from tempor.plugins.pipeline import Pipeline
from tempor.utils.dataloaders import (
    GoogleStocksDataLoader,
    PBCDataLoader,
    SineDataLoader,
)


@pytest.mark.parametrize(
    "model_template",
    [
        Pipeline(
            [
                "preprocessing.imputation.ffill",
                "classification.nn_classifier",
            ]
        )({"nn_classifier": {"n_iter": 50}}),
        Pipeline(
            [
                "classification.nn_classifier",
            ]
        )({"nn_classifier": {"n_iter": 50}}),
        plugin_loader.get("classification.nn_classifier", n_iter=50),
    ],
)
@pytest.mark.parametrize("n_splits", [2])
def test_classifier_evaluation(model_template: Any, n_splits: int) -> None:
    dataset = SineDataLoader().load()

    scores = evaluate_classifier(model_template, dataset, n_splits=n_splits, seed=0)

    for out_metric in output_metrics:
        assert out_metric in scores

    for metric in classifier_supported_metrics:
        assert metric in scores.index


@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_classifier_evaluation_fail(n_splits: int) -> None:
    dataset = SineDataLoader().load()

    with pytest.raises(ValueError):
        evaluate_classifier(
            plugin_loader.get("classification.nn_classifier", n_iter=50), dataset, n_splits=n_splits, seed=0
        )


@pytest.mark.parametrize(
    "model_template",
    [
        Pipeline(
            [
                "preprocessing.imputation.ffill",
                "regression.nn_regressor",
            ]
        )({"nn_regressor": {"n_iter": 50}}),
        Pipeline(
            [
                "regression.nn_regressor",
            ]
        )({"nn_regressor": {"n_iter": 50}}),
        plugin_loader.get("regression.nn_regressor", n_iter=50),
    ],
)
@pytest.mark.parametrize("n_splits", [2])
def test_regressor_evaluation(model_template: Any, n_splits: int) -> None:
    dataset = GoogleStocksDataLoader().load()

    scores = evaluate_regressor(model_template, dataset, n_splits=n_splits, seed=0)

    for out_metric in output_metrics:
        assert out_metric in scores

    for metric in regression_supported_metrics:
        assert metric in scores.index


@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_regressor_evaluation_fail(n_splits: int) -> None:
    dataset = SineDataLoader().load()

    with pytest.raises(ValueError):
        evaluate_regressor(plugin_loader.get("regression.nn_regressor", n_iter=50), dataset, n_splits=n_splits, seed=0)


@pytest.mark.parametrize(
    "model_template",
    [
        Pipeline(
            [
                "preprocessing.imputation.ffill",
                "time_to_event.dynamic_deephit",
            ]
        )({"dynamic_deephit": {"n_iter": 10}}),
        Pipeline(
            [
                "time_to_event.dynamic_deephit",
            ]
        )({"dynamic_deephit": {"n_iter": 10}}),
        plugin_loader.get("time_to_event.dynamic_deephit", n_iter=10),
    ],
)
@pytest.mark.parametrize("n_splits", [2])
def test_time_to_event_evaluation(model_template: Any, n_splits: int, get_event0_time_percentiles: Callable) -> None:
    dataset = PBCDataLoader().load()

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    scores = evaluate_time_to_event(model_template, dataset, n_splits=n_splits, seed=0, horizons=horizons)

    for out_metric in output_metrics:
        assert out_metric in scores

    for metric in time_to_event_supported_metrics:
        assert metric in scores.index


@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_time_to_event_evaluation_fail(n_splits: int, get_event0_time_percentiles: Callable) -> None:
    dataset = PBCDataLoader().load()

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    with pytest.raises(ValueError):
        evaluate_time_to_event(
            plugin_loader.get("time_to_event.nn_regressor", n_iter=50),
            dataset,
            n_splits=n_splits,
            seed=0,
            horizons=horizons,
        )
