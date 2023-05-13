# pylint: disable=redefined-outer-name,unused-argument

from typing import Callable
from unittest.mock import MagicMock

import pytest

from tempor.automl.seeker import METRIC_DIRECTION_MAP, SimpleSeeker, TunerType
from tempor.data.dataset import PredictiveDataset, TimeToEventAnalysisDataset
from tempor.plugins.core._params import CategoricalParams, IntegerParams


@pytest.fixture
def patch_slow(monkeypatch):
    # Monkeypatch all the slow things for ease of unit testing, specifically:
    #   - Evaluation functions.
    #   - Estimator methods fit/predict etc.

    import numpy as np
    import pandas as pd

    from tempor.benchmarks import evaluation
    from tempor.plugins.core import BasePredictor

    def patched_fit(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self

    def patched_predict(self, *args, **kwargs):  # pylint: disable=unused-argument
        return MagicMock()

    monkeypatch.setattr(BasePredictor, "fit", patched_fit)
    monkeypatch.setattr(BasePredictor, "predict", patched_predict)

    np.random.seed(12345)

    def patched_evaluate(*args, **kwargs):
        return pd.DataFrame(
            data=np.random.rand(len(METRIC_DIRECTION_MAP), 1),
            index=list(METRIC_DIRECTION_MAP.keys()),
            columns=["mean"],
        )

    monkeypatch.setattr(evaluation, "evaluate_prediction_oneoff_classifier", patched_evaluate)
    monkeypatch.setattr(evaluation, "evaluate_prediction_oneoff_regressor", patched_evaluate)
    monkeypatch.setattr(evaluation, "evaluate_time_to_event", patched_evaluate)


class TestSimpleSeeker:
    @pytest.mark.parametrize("tuner_type", ["bayesian", "random", "cmaes", "qmc"])
    @pytest.mark.parametrize(
        "override_hp_space",
        [
            None,
            {
                "cde_classifier": [
                    IntegerParams(name="n_iter", low=2, high=100),
                ],
                "ode_classifier": [
                    IntegerParams(name="n_iter", low=2, high=100),
                ],
                "nn_classifier": [
                    IntegerParams(name="n_iter", low=2, high=100),
                ],
            },
        ],
    )
    def test_init_success(self, tuner_type: TunerType, override_hp_space, get_dataset: Callable):
        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        dataset = get_dataset("sine_data_full")
        SimpleSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            tuner_type=tuner_type,
            override_hp_space=override_hp_space,
        )

    def test_init_fails_wrong_overrides(self, get_dataset: Callable):
        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        override_hp_space = {
            "cde_classifier": [
                IntegerParams(name="n_iter", low=2, high=100),
            ],
            "nonexistent_classifier": [
                IntegerParams(name="n_iter", low=2, high=100),
            ],
        }
        dataset = get_dataset("sine_data_full")

        with pytest.raises(ValueError, match=".*override.*did not correspond.*"):
            SimpleSeeker(
                study_name="test_study",
                task_type="prediction.one_off.classification",
                estimator_names=estimator_names,
                metric="aucroc",
                dataset=dataset,
                tuner_type="bayesian",
                override_hp_space=override_hp_space,  # type: ignore
            )

    def test_init_success_grid(self, get_dataset: Callable):
        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        grid = {
            "cde_classifier": {"n_iter": [10, 100], "lr": [0.01, 0.001]},
            "ode_classifier": {"n_iter": [10, 100], "lr": [0.01, 0.001]},
            "nn_classifier": {"n_iter": [10, 100], "lr": [0.01, 0.001]},
        }
        dataset = get_dataset("sine_data_full")
        SimpleSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            tuner_type="grid",
            grid=grid,
        )

    def test_init_fails_grid(self, get_dataset: Callable):
        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        wrong_grid = {
            "ode_classifier": {"n_iter": [10, 100], "lr": [0.01, 0.001]},
            "mm_classifier": {"n_iter": [10, 100], "lr": [0.01, 0.001]},
        }
        dataset = get_dataset("sine_data_full")
        with pytest.raises(ValueError, match=".*[Mm]ust provide.*grid.*"):
            SimpleSeeker(
                study_name="test_study",
                task_type="prediction.one_off.classification",
                estimator_names=estimator_names,
                metric="aucroc",
                dataset=dataset,
                tuner_type="grid",
                grid=None,
            )
        with pytest.raises(ValueError, match=".*grid.*did not match.*"):
            SimpleSeeker(
                study_name="test_study",
                task_type="prediction.one_off.classification",
                estimator_names=estimator_names,
                metric="aucroc",
                dataset=dataset,
                tuner_type="grid",
                grid=wrong_grid,
            )

    @pytest.mark.parametrize(
        "task_type,estimator_names,metric",
        [
            (
                "prediction.one_off.classification",
                [
                    "cde_classifier",
                    "ode_classifier",
                    "nn_classifier",
                    "laplace_ode_classifier",
                ],
                "aucroc",
            ),
            (
                "prediction.one_off.regression",
                [
                    "cde_regressor",
                    "ode_regressor",
                    "nn_regressor",
                    "laplace_ode_regressor",
                ],
                "mse",
            ),
            (
                "time_to_event",
                [
                    "dynamic_deephit",
                    "ts_coxph",
                    "ts_xgb",
                ],
                "c_index",
            ),
        ],
    )
    @pytest.mark.parametrize("override", [False, True])
    def test_search(self, task_type, estimator_names, metric, override, patch_slow):
        if task_type != "time_to_event":
            dataset = MagicMock(PredictiveDataset)
        else:
            dataset = MagicMock(TimeToEventAnalysisDataset)

        horizon = None
        if task_type == "time_to_event":
            horizon = [0.1, 0.5, 0.7]

        override_hp_space = None
        if override:
            override_hp_space = {estimator_names[0]: [CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4])]}

        seeker = SimpleSeeker(
            study_name="test_study",
            task_type=task_type,
            estimator_names=estimator_names,
            metric=metric,
            dataset=dataset,
            return_top_k=3,
            num_iter=10,
            override_hp_space=override_hp_space,  # type: ignore
            horizon=horizon,
        )

        estimators, scores = seeker.search()

        assert len(estimators) == len(scores) == 3
        for est in estimators:
            assert est.name in estimator_names
        if seeker.direction == "maximize":
            assert sorted(scores, reverse=True) == scores
        else:
            assert sorted(scores, reverse=False) == scores

    def test_search_fail_time_to_event(self, patch_slow):
        task_type = "time_to_event"
        estimator_names = [
            "dynamic_deephit",
            "ts_coxph",
            "ts_xgb",
        ]

        # No horizon.
        seeker = SimpleSeeker(
            study_name="test_study",
            task_type=task_type,
            estimator_names=estimator_names,
            metric="c_index",
            dataset=MagicMock(TimeToEventAnalysisDataset),
            return_top_k=3,
            num_iter=10,
            horizon=None,
        )
        with pytest.raises(ValueError, match=".*horizon.*"):
            seeker.search()

        # Wrong dataset.
        seeker = SimpleSeeker(
            study_name="test_study",
            task_type=task_type,
            estimator_names=estimator_names,
            metric="c_index",
            dataset=MagicMock(PredictiveDataset),
            return_top_k=3,
            num_iter=10,
            horizon=[0.1, 0.4, 0.9],
        )
        with pytest.raises(ValueError, match=".*dataset.*"):
            seeker.search()

    # TODO: May wish to add more cases of slow tests, mark them as extra.
    @pytest.mark.slow
    def test_search_end2end(self, get_dataset: Callable):
        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        override_hp_space = {
            "cde_classifier": [
                IntegerParams(name="n_iter", low=2, high=2),
                IntegerParams(name="n_temporal_units_hidden", low=5, high=20),
                CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
            ],
            "ode_classifier": [
                IntegerParams(name="n_iter", low=2, high=2),
                IntegerParams(name="n_units_hidden", low=5, high=20),
                CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
            ],
            "nn_classifier": [
                IntegerParams(name="n_iter", low=2, high=2),
                IntegerParams(name="n_units_hidden", low=5, high=20),
                CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
            ],
        }
        dataset = get_dataset("sine_data_full")

        seeker = SimpleSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            return_top_k=2,
            num_iter=3,
            override_hp_space=override_hp_space,  # type: ignore
        )

        estimators, scores = seeker.search()

        assert len(estimators) == len(scores) == 2
        for est in estimators:
            assert est.name in estimator_names
