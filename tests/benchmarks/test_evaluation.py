from typing import Any, Callable
from unittest.mock import Mock

import pytest

from tempor.benchmarks import (
    classifier_supported_metrics,
    evaluate_prediction_oneoff_classifier,
    evaluate_prediction_oneoff_regressor,
    evaluate_time_to_event,
    output_metrics,
    regression_supported_metrics,
    time_to_event_supported_metrics,
)
from tempor.benchmarks.evaluation import ClassifierMetrics
from tempor.plugins import plugin_loader
from tempor.plugins.pipeline import pipeline

N_ITER = 5

TEST_ON_DATASETS_CLASSIFIER = ["sine_data_small"]
TEST_ON_DATASETS_REGRESSOR = ["google_stocks_data_small"]
TEST_ON_DATASETS_TIME_TO_EVENT = ["pbc_data_small"]

PREDICTOR_CLASSIFICATION = "prediction.one_off.classification.nn_classifier"
PREDICTOR_REGRESSION = "prediction.one_off.regression.nn_regressor"
PREDICTOR_TIME_TO_EVENT = "time_to_event.dynamic_deephit"


def test_classifier_metrics_score_proba_input_validation():
    clf_metrics = ClassifierMetrics()
    with pytest.raises(ValueError, match=".*.input*"):
        clf_metrics.score_proba(None, None)  # type: ignore


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
def test_evaluate_prediction_oneoff_classifier(
    model_template: Any, n_splits: int, data: str, request: pytest.FixtureRequest
) -> None:
    dataset = request.getfixturevalue(data)

    scores = evaluate_prediction_oneoff_classifier(model_template, dataset, n_splits=n_splits, seed=0)

    for out_metric in output_metrics:
        assert out_metric in scores

    for metric in classifier_supported_metrics:
        assert metric in scores.index


@pytest.mark.parametrize("data", TEST_ON_DATASETS_CLASSIFIER)
@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_evaluate_prediction_oneoff_classifier_fail(n_splits: int, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    with pytest.raises(ValueError):
        evaluate_prediction_oneoff_classifier(
            plugin_loader.get(PREDICTOR_CLASSIFICATION, n_iter=N_ITER), dataset, n_splits=n_splits, seed=0
        )


def test_evaluate_prediction_oneoff_classifier_fail_validation() -> None:
    import numpy as np

    from tempor.data.dataset import PredictiveDataset

    mock_data = Mock(PredictiveDataset, predictive=Mock(targets=None), num_features=1)

    with pytest.raises(ValueError, match=".*targets.*"):
        evaluate_prediction_oneoff_classifier(
            plugin_loader.get(PREDICTOR_CLASSIFICATION, n_iter=N_ITER), mock_data, n_splits=3, seed=0
        )

    mock_data = Mock(
        PredictiveDataset, predictive=Mock(targets=Mock(numpy=Mock(return_value=np.ones(shape=(5, 5)))), num_features=1)
    )

    with pytest.raises(ValueError, match=".*1D.*"):
        evaluate_prediction_oneoff_classifier(
            plugin_loader.get(PREDICTOR_CLASSIFICATION, n_iter=N_ITER), mock_data, n_splits=3, seed=0
        )


@pytest.mark.parametrize("data", TEST_ON_DATASETS_CLASSIFIER)
@pytest.mark.parametrize("do_raise", [True, False])
def test_evaluate_prediction_oneoff_classifier_model_error(
    data: str, request: pytest.FixtureRequest, do_raise: bool, monkeypatch
):
    dataset = request.getfixturevalue(data)
    p = plugin_loader.get(PREDICTOR_CLASSIFICATION, n_iter=N_ITER)

    def raise_(*args, **kwargs):
        raise ValueError("test error")

    monkeypatch.setattr(p, "fit", raise_)

    if do_raise:
        with pytest.raises(ValueError, match=".*test error.*"):
            evaluate_prediction_oneoff_classifier(
                p,
                dataset,
                n_splits=3,
                seed=0,
                raise_exceptions=do_raise,
            )
    else:
        scores = evaluate_prediction_oneoff_classifier(
            p,
            dataset,
            n_splits=3,
            seed=0,
            raise_exceptions=do_raise,
        )
        assert (scores["errors"] > 0).all()


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
def test_evaluate_prediction_oneoff_regressor(
    model_template: Any, n_splits: int, data: str, request: pytest.FixtureRequest
) -> None:
    dataset = request.getfixturevalue(data)

    scores = evaluate_prediction_oneoff_regressor(model_template, dataset, n_splits=n_splits, seed=0)

    for out_metric in output_metrics:
        assert out_metric in scores

    for metric in regression_supported_metrics:
        assert metric in scores.index


@pytest.mark.parametrize("data", TEST_ON_DATASETS_REGRESSOR)
@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_evaluate_prediction_oneoff_regressor_fail(n_splits: int, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    with pytest.raises(ValueError):
        evaluate_prediction_oneoff_regressor(
            plugin_loader.get(PREDICTOR_REGRESSION, n_iter=N_ITER), dataset, n_splits=n_splits, seed=0
        )


@pytest.mark.parametrize("data", TEST_ON_DATASETS_REGRESSOR)
@pytest.mark.parametrize("do_raise", [True, False])
def test_evaluate_prediction_oneoff_regressor_error(
    data: str, request: pytest.FixtureRequest, do_raise: bool, monkeypatch
):
    dataset = request.getfixturevalue(data)
    p = plugin_loader.get(PREDICTOR_REGRESSION, n_iter=N_ITER)

    def raise_(*args, **kwargs):
        raise ValueError("test error")

    monkeypatch.setattr(p, "fit", raise_)

    if do_raise:
        with pytest.raises(ValueError, match=".*test error.*"):
            evaluate_prediction_oneoff_regressor(
                p,
                dataset,
                n_splits=3,
                seed=0,
                raise_exceptions=do_raise,
            )
    else:
        scores = evaluate_prediction_oneoff_regressor(
            p,
            dataset,
            n_splits=3,
            seed=0,
            raise_exceptions=do_raise,
        )
        assert (scores["errors"] > 0).all()


@pytest.mark.filterwarnings("ignore:.*Validation.*small.*:RuntimeWarning")  # Expected for small test datasets with DDH.
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
            plugin_loader.get("time_to_event.dynamic_deephit", n_iter=N_ITER),
            dataset,
            n_splits=n_splits,
            seed=0,
            horizons=horizons,
        )


@pytest.mark.parametrize("data", TEST_ON_DATASETS_TIME_TO_EVENT)
@pytest.mark.parametrize("do_raise", [True, False])
def test_time_to_event_model_error(
    get_event0_time_percentiles: Callable, data: str, request: pytest.FixtureRequest, do_raise: bool, monkeypatch
):
    dataset = request.getfixturevalue(data)
    p = plugin_loader.get(PREDICTOR_TIME_TO_EVENT, n_iter=N_ITER)

    def raise_(*args, **kwargs):
        raise ValueError("test error")

    monkeypatch.setattr(p, "fit", raise_)

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    if do_raise:
        with pytest.raises(ValueError, match=".*test error.*"):
            evaluate_time_to_event(
                p,
                dataset,
                n_splits=3,
                seed=0,
                horizons=horizons,
                raise_exceptions=do_raise,
            )
    else:
        scores = evaluate_time_to_event(
            p,
            dataset,
            n_splits=3,
            seed=0,
            horizons=horizons,
            raise_exceptions=do_raise,
        )
        assert (scores["errors"] > 0).all()


def test_compute_time_to_event_metric():
    import pandas as pd

    from tempor.benchmarks.evaluation import _compute_time_to_event_metric

    metric_func = Mock()
    mock_pred = Mock(num_timesteps_equal=Mock(return_value=True), num_features=1)
    mock_test_data = Mock(predictive=Mock(targets=Mock(num_features=1)))
    mock_train_data = Mock(predictive=Mock(targets=Mock(num_features=1)))

    with pytest.raises(ValueError, match=".*horizon.*"):
        _compute_time_to_event_metric(
            metric_func=metric_func,
            train_data=mock_train_data,
            test_data=mock_test_data,
            horizons=pd.to_datetime(["2000-01-01", "2000-01-02"]),  # type: ignore
            predictions=mock_pred,
        )

    with pytest.raises(ValueError, match=".*one event.*"):
        _compute_time_to_event_metric(
            metric_func=metric_func,
            train_data=Mock(predictive=Mock(targets=Mock(num_features=999))),
            test_data=mock_test_data,
            horizons=Mock(),
            predictions=mock_pred,
        )

    with pytest.raises(ValueError, match=".*targets.*"):
        _compute_time_to_event_metric(
            metric_func=metric_func,
            train_data=Mock(predictive=Mock(targets=None)),
            test_data=mock_test_data,
            horizons=Mock(),
            predictions=mock_pred,
        )

    with pytest.raises(ValueError, match=".*equal number of time steps.*"):
        _compute_time_to_event_metric(
            metric_func=metric_func,
            train_data=mock_train_data,
            test_data=mock_test_data,
            horizons=[1, 2, 3],
            predictions=Mock(num_timesteps_equal=Mock(return_value=False)),
        )
