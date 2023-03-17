from typing import Any

import pytest

from tempor.benchmarks.evaluation import (
    classifier_supported_metrics,
    evaluate_classifier,
    evaluate_regression,
    regression_supported_metrics,
)
from tempor.plugins import plugin_loader
from tempor.plugins.pipeline import Pipeline
from tempor.utils.datasets.google_stocks import GoogleStocksDataloader
from tempor.utils.datasets.sine import SineDataloader


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
    dataset = SineDataloader().load()

    scores = evaluate_classifier(model_template, dataset, n_splits=n_splits, seed=0)

    assert "str" in scores
    assert "raw" in scores

    for metric in classifier_supported_metrics:
        assert metric in scores["str"]
        assert metric in scores["raw"]


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
    dataset = GoogleStocksDataloader().load()

    scores = evaluate_regression(model_template, dataset, n_splits=n_splits, seed=0)

    assert "str" in scores
    assert "raw" in scores

    for metric in regression_supported_metrics:
        assert metric in scores["str"]
        assert metric in scores["raw"]
