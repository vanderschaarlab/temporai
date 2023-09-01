# pylint: disable=redefined-outer-name,unused-argument

from typing import Any, Callable, Dict
from unittest.mock import MagicMock

import pytest

from tempor.automl._types import AutoMLCompatibleEstimator
from tempor.automl.seeker import (
    DEFAULT_STATIC_SCALERS,
    DEFAULT_TEMPORAL_SCALERS,
    METRIC_DIRECTION_MAP,
    BaseSeeker,
    MethodSeeker,
    PipelineSeeker,
    TunerType,
)
from tempor.data.dataset import PredictiveDataset, TimeToEventAnalysisDataset
from tempor.plugins.core._params import CategoricalParams, IntegerParams


def test_init_fails_estimator_name_def_length_mismatch():
    class MySeeker(BaseSeeker):
        def __init__(self, **kwargs) -> None:
            super().__init__(
                study_name="test",
                task_type="prediction.one_off.classification",
                estimator_names=["a", "b"],
                estimator_defs=["a", "b", "c"],
                metric="accuracy",
                dataset=MagicMock(PredictiveDataset),
                **kwargs,
            )

        def _init_estimator(self, estimator_name: str, estimator_def: Any):
            pass

        def _create_estimator_with_hps(
            self, estimator_cls: AutoMLCompatibleEstimator, hps: Dict[str, Any], score: float
        ):
            pass

    with pytest.raises(ValueError, match=".*length.*"):
        MySeeker()


@pytest.fixture
def patch_slow(monkeypatch, request):
    # Monkeypatch all the slow things for ease of unit testing, specifically:
    #   - Evaluation functions.
    #   - Estimator methods fit/predict etc.
    #   - Pipeline method generation fit/predict etc.

    import numpy as np
    import pandas as pd

    from tempor.benchmarks import evaluation
    from tempor.plugins.core import BasePredictor, BaseTransformer

    def patched_fit(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self

    def patched_predict(self, *args, **kwargs):  # pylint: disable=unused-argument
        return MagicMock()

    def patched_transform(self, *args, **kwargs):  # pylint: disable=unused-argument
        return MagicMock()

    monkeypatch.setattr(BasePredictor, "fit", patched_fit)
    monkeypatch.setattr(BasePredictor, "predict", patched_predict)
    monkeypatch.setattr(BaseTransformer, "transform", patched_transform)

    np.random.seed(12345)

    def patched_evaluate(*args, **kwargs):
        # Since "ensure reproducibility" may affect seeding, and we want this function to return different values,
        # seed it from input hash manually as below.
        seed_from = str(args[0])
        hash_ = hash(seed_from)
        # Ensure fits with random seed requirements
        hash_ = -1 * hash_ if hash_ < 0 else hash_
        hash_ = int(str(hash_)[-9:])
        np.random.seed(hash_)
        # --- --- ---

        return pd.DataFrame(
            data=np.random.rand(len(METRIC_DIRECTION_MAP), 1),
            index=list(METRIC_DIRECTION_MAP.keys()),
            columns=["mean"],
        )

    monkeypatch.setattr(evaluation, "evaluate_prediction_oneoff_classifier", patched_evaluate)
    monkeypatch.setattr(evaluation, "evaluate_prediction_oneoff_regressor", patched_evaluate)
    monkeypatch.setattr(evaluation, "evaluate_time_to_event", patched_evaluate)

    from tempor.plugins.pipeline import generators

    def patched_generate_fit():
        return patched_fit

    def patched_generate_predict():
        return patched_predict

    monkeypatch.setattr(generators, "_generate_fit", patched_generate_fit)
    monkeypatch.setattr(generators, "_generate_predict", patched_generate_predict)


class TestMethodSeeker:
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
        MethodSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            tuner_type=tuner_type,
            override_hp_space=override_hp_space,
        )

    def test_init_success_custom_tuner(self, get_dataset: Callable):
        import optuna

        from tempor.automl.tuner import OptunaTuner

        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        dataset = get_dataset("sine_data_full")

        custom_tuner = OptunaTuner(
            study_name="test_study",
            direction="maximize",
            study_sampler=optuna.samplers.TPESampler(seed=12345),
            study_pruner=optuna.pruners.PercentilePruner(0.1),
        )

        seeker = MethodSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            override_hp_space=None,
            custom_tuner=custom_tuner,
        )
        assert len(seeker.tuners) == len(estimator_names)
        for tuner in seeker.tuners:
            assert isinstance(tuner, OptunaTuner)
            assert "test_study_" in tuner.study_name
            isinstance(tuner.study_pruner, optuna.pruners.PercentilePruner)

    def test_init_fails_custom_tuner_grid_sampler_unsupported(self, get_dataset: Callable):
        import optuna

        from tempor.automl.tuner import OptunaTuner

        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        dataset = get_dataset("sine_data_full")

        custom_tuner = OptunaTuner(
            study_name="test_study",
            direction="maximize",
            study_sampler=optuna.samplers.GridSampler(search_space={}),
            study_pruner=optuna.pruners.PercentilePruner(0.1),
        )

        with pytest.raises(ValueError, match=".*[Gg]rid.*not supported.*"):
            MethodSeeker(
                study_name="test_study",
                task_type="prediction.one_off.classification",
                estimator_names=estimator_names,
                metric="aucroc",
                dataset=dataset,
                override_hp_space=None,
                custom_tuner=custom_tuner,
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
            MethodSeeker(
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
        MethodSeeker(
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
            MethodSeeker(
                study_name="test_study",
                task_type="prediction.one_off.classification",
                estimator_names=estimator_names,
                metric="aucroc",
                dataset=dataset,
                tuner_type="grid",
                grid=None,
            )
        with pytest.raises(ValueError, match=".*grid.*did not match.*"):
            MethodSeeker(
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

        seeker = MethodSeeker(
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
        seeker = MethodSeeker(
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
        seeker = MethodSeeker(
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

        seeker = MethodSeeker(
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

    @pytest.mark.slow
    def test_search_end2end_custom_tuner(self, get_dataset: Callable):
        import optuna

        from tempor.automl.tuner import OptunaTuner

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

        custom_tuner = OptunaTuner(
            study_name="test_study",
            direction="maximize",
            study_sampler=optuna.samplers.TPESampler(seed=12345),
            study_pruner=optuna.pruners.PercentilePruner(0.1),
        )

        seeker = MethodSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            return_top_k=2,
            num_iter=3,
            override_hp_space=override_hp_space,  # type: ignore
            custom_tuner=custom_tuner,
        )

        estimators, scores = seeker.search()

        assert len(estimators) == len(scores) == 2
        for est in estimators:
            assert est.name in estimator_names


class TestPipelineSeeker:
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
        PipelineSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            tuner_type=tuner_type,
            override_hp_space=override_hp_space,
        )

    def test_init_success_custom_tuner(self, get_dataset: Callable):
        import optuna

        from tempor.automl.tuner import OptunaTuner

        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        dataset = get_dataset("sine_data_full")

        custom_tuner = OptunaTuner(
            study_name="test_study",
            direction="maximize",
            study_sampler=optuna.samplers.TPESampler(seed=12345),
            study_pruner=optuna.pruners.PercentilePruner(0.1),
        )

        seeker = PipelineSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            override_hp_space=None,
            custom_tuner=custom_tuner,
        )
        assert len(seeker.tuners) == len(estimator_names)
        for tuner in seeker.tuners:
            assert isinstance(tuner, OptunaTuner)
            assert "test_study_" in tuner.study_name
            isinstance(tuner.study_pruner, optuna.pruners.PercentilePruner)

    def test_init_fails_grid_unsupported(self, get_dataset: Callable):
        estimator_names = [
            "cde_classifier",
            "ode_classifier",
            "nn_classifier",
        ]
        dataset = get_dataset("sine_data_full")

        with pytest.raises(ValueError, match=".*[Gg]rid.*not supported.*"):
            PipelineSeeker(
                study_name="test_study",
                task_type="prediction.one_off.classification",
                estimator_names=estimator_names,
                metric="aucroc",
                dataset=dataset,
                override_hp_space=None,
                tuner_type="grid",
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
            PipelineSeeker(
                study_name="test_study",
                task_type="prediction.one_off.classification",
                estimator_names=estimator_names,
                metric="aucroc",
                dataset=dataset,
                tuner_type="bayesian",
                override_hp_space=override_hp_space,  # type: ignore
            )

    @pytest.mark.filterwarnings("ignore:.*bool8.*:DeprecationWarning")  # Expected.
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

        seeker = PipelineSeeker(
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

        pipelines, scores = seeker.search()

        assert len(pipelines) == len(scores) == 3
        for pipe in pipelines:
            assert pipe.stages[-1].name in estimator_names  # type: ignore
        if seeker.direction == "maximize":
            assert sorted(scores, reverse=True) == scores
        else:
            assert sorted(scores, reverse=False) == scores

    @pytest.mark.slow
    def test_search_end2end(self, get_dataset: Callable):
        from tempor.plugins.pipeline import PipelineBase

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

        seeker = PipelineSeeker(
            study_name="test_study",
            task_type="prediction.one_off.classification",
            estimator_names=estimator_names,
            metric="aucroc",
            dataset=dataset,
            return_top_k=2,
            num_iter=3,
            override_hp_space=override_hp_space,  # type: ignore
            static_imputers=[],  # Skip, slow.
            static_scalers=DEFAULT_STATIC_SCALERS,
            temporal_imputers=[],  # Skip, slow.
            temporal_scalers=DEFAULT_TEMPORAL_SCALERS,
        )

        pipelines, scores = seeker.search()

        assert len(pipelines) == len(scores) == 2
        for pipe in pipelines:
            assert isinstance(pipe, PipelineBase)
            assert len(pipe.stages) == 3
            assert pipe.stages[-1].name in estimator_names
            assert pipe.stages[0].name in DEFAULT_STATIC_SCALERS
            assert pipe.stages[1].name in DEFAULT_TEMPORAL_SCALERS

            # Assert override workedL
            assert pipe.stages[-1].params.n_iter == 2
