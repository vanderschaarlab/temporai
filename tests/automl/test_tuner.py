# pylint: disable=redefined-outer-name

import functools
import warnings
from typing import Any, Callable, Dict, Type
from unittest.mock import Mock

import optuna
import pytest
from packaging.version import Version

from tempor.automl import OptimDirection, tuner
from tempor.benchmarks import evaluation
from tempor.data.dataset import PredictiveDataset, TimeToEventAnalysisDataset
from tempor.methods import plugin_loader
from tempor.methods.core._base_predictor import BasePredictor
from tempor.methods.core._params import IntegerParams

# To ignore warnings in parametrization:
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


OPTIMIZE_KWARGS = {"n_trials": 2}

# When testing optuna.samplers.GridSampler, build the grid from this many calls to sample_hyperparameters:
GRID_SAMPLER_GRID_N = 2

# The number or iters/epochs of underlying models is limited in the tests for the sake of speed.
# See `limit_int_param()`.
MIN_MODEL_EPOCHS = 2
MAX_MODEL_EPOCHS = 3

SEED = 123
SETTINGS: Dict = dict()

TEST_SAMPLER_SET = [
    optuna.samplers.TPESampler(seed=SEED),
    pytest.param(optuna.samplers.RandomSampler(seed=SEED), marks=pytest.mark.extra),
    pytest.param(optuna.samplers.CmaEsSampler(seed=SEED), marks=pytest.mark.extra),
    pytest.param(
        optuna.samplers.QMCSampler(seed=SEED),
        marks=[
            pytest.mark.skipif(
                Version(optuna.__version__) < Version("3.0.0"),
                reason="optuna.samplers.QMCSampler is only available in optuna v3.0.0+",
            ),
            pytest.mark.extra,
        ],
    ),
    pytest.param("GridSampler", marks=pytest.mark.extra),
    # ^ Needs get_grid_by_sampling() call in test to define the grid.
    # NOTE: `optuna.samplers.BruteForceSampler` not currently compatible due to FloatParams
    # infinite search space; MO samplers are not applicable.
]
TEST_PRUNER_SET = [
    optuna.pruners.MedianPruner(),
    pytest.param(optuna.pruners.PatientPruner(None, 3), marks=pytest.mark.extra),
    pytest.param(optuna.pruners.PercentilePruner(0.25), marks=pytest.mark.extra),
    pytest.param(optuna.pruners.SuccessiveHalvingPruner(), marks=pytest.mark.extra),
    pytest.param(optuna.pruners.ThresholdPruner(lower=0.5), marks=pytest.mark.extra),
    pytest.param(optuna.pruners.HyperbandPruner(), marks=pytest.mark.extra),
]


# Test settings for different evaluation cases. ---
_category_prefix = "prediction.one_off.classification"
SETTINGS["prediction.one_off.classification"] = dict(
    TEST_ON_DATASETS=["sine_data_small"],
    TEST_WITH_PLUGINS=[
        f"{_category_prefix}.nn_classifier",
        pytest.param(f"{_category_prefix}.cde_classifier", marks=pytest.mark.extra),
        pytest.param(f"{_category_prefix}.ode_classifier", marks=pytest.mark.extra),
    ],
    METRICS=[
        ("aucroc", "maximize"),
        ("accuracy", "maximize"),
        ("precision_micro", "maximize"),
    ],
    TEST_SAMPLER_SET=TEST_SAMPLER_SET,
)
_category_prefix = "prediction.one_off.regression"
SETTINGS["prediction.one_off.regression"] = dict(
    TEST_ON_DATASETS=["sine_data_small"],
    TEST_WITH_PLUGINS=[
        f"{_category_prefix}.nn_regressor",
        pytest.param(f"{_category_prefix}.cde_regressor", marks=pytest.mark.extra),
        pytest.param(f"{_category_prefix}.ode_regressor", marks=pytest.mark.extra),
    ],
    METRICS=[
        ("mse", "minimize"),
        ("mae", "minimize"),
        ("r2", "maximize"),
    ],
    TEST_SAMPLER_SET=TEST_SAMPLER_SET,
)
_category_prefix = "time_to_event"
SETTINGS["time_to_event"] = dict(
    TEST_ON_DATASETS=["pbc_data_small"],
    TEST_WITH_PLUGINS=[
        f"{_category_prefix}.dynamic_deephit",
        pytest.param(f"{_category_prefix}.ts_coxph", marks=pytest.mark.extra),
        pytest.param(
            f"{_category_prefix}.ts_xgb",
            marks=[
                pytest.mark.extra,
                pytest.mark.skipci,  # Too memory intensive for CI workers, skip.
            ],
        ),
    ],
    METRICS=[
        ("c_index", "maximize"),
        ("brier_score", "minimize"),
    ],
    TEST_SAMPLER_SET=[s for s in TEST_SAMPLER_SET if s != "GridSampler"],  # GridSampler too resource intensive.
)
# Test settings for different evaluation cases. (end) ---


def get_grid_by_sampling(plugin_cls: Any, n: int) -> Dict:
    # Get a parameter grid for `plugin_cls` like {"param1": [val1, val2, val3], "param2": [val1, val2, val3]}
    # by calling `plugin_cls.sample_hyperparameters()` `n` times.
    out: Dict = dict()
    for i in range(n):
        hps = plugin_cls.sample_hyperparameters()
        for k, v in hps.items():
            if i == 0:
                out[k] = [v]
            else:
                out[k].append(v)
    return out


@pytest.fixture
def limit_int_param(monkeypatch):
    # A fixture that overrides `hyperparameter_space` method for a plugin to limit the `param_name` parameter space.
    # This is used to speed up tests by limiting epoch params, like: ``n_iter``, ``epochs``.
    def func(plugin: Any, param_name: str, low: int, high: int):
        original = plugin.hyperparameter_space

        def patched_hyperparameter_space(*args, **kwargs):
            hps = original(*args, **kwargs)
            hps = [hp for hp in hps if hp.name != param_name]
            hps.append(
                IntegerParams(name=param_name, low=low, high=high),
            )
            return hps

        monkeypatch.setattr(plugin, "hyperparameter_space", patched_hyperparameter_space)
        return plugin

    return func


@pytest.fixture
def tune_objective(get_event0_time_percentiles: Callable):
    # The tuning objective function.
    # We take the appropriate function from the tempor.benchmarks.evaluation module.

    def func(
        estimator: Type[BasePredictor],
        dataset: PredictiveDataset,
        evaluation_case: str,
        metric: str,
        *args,
        **kwargs,
    ) -> float:
        model = estimator(*args, **kwargs)
        # TODO: Handle missing cases.
        if evaluation_case == "prediction.one_off.classification":
            metrics = evaluation.evaluate_prediction_oneoff_classifier(model, dataset)
        elif evaluation_case == "prediction.one_off.regression":
            metrics = evaluation.evaluate_prediction_oneoff_regressor(model, dataset)
        elif evaluation_case == "prediction.temporal.classification":
            raise NotImplementedError
        elif evaluation_case == "prediction.temporal.regression":
            raise NotImplementedError
        elif evaluation_case == "time_to_event":
            assert isinstance(dataset, TimeToEventAnalysisDataset)
            metrics = evaluation.evaluate_time_to_event(
                model, dataset, horizons=get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])
            )
        elif evaluation_case == "treatments.one_off.classification":
            raise NotImplementedError
        elif evaluation_case == "treatments.one_off.regression":
            raise NotImplementedError
        elif evaluation_case == "treatments.temporal.classification":
            raise NotImplementedError
        elif evaluation_case == "treatments.temporal.regression":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown evaluation case: {evaluation_case}")
        return metrics.loc[metric, "mean"]  # pyright: ignore

    return func


# TestOptunaTuner helper functions. ---


@pytest.fixture
def helper_initialize(limit_int_param: Callable, get_dataset: Callable):
    def func(data: str, plugin: str):
        # Initialize the dataset and plugin class from the strings provided.
        # In addition, limit the `n_iter`, `epochs` hyperparameter sampling, such that tests run in reasonable time.

        dataset = get_dataset(data)
        p = plugin_loader.get_class(plugin)

        # NOTE: Limit the hyperparameters like "n_iter" and "epochs", so that tests do not take a very long time.
        for param_name in ("n_iter", "epochs"):
            if hasattr(p.ParamsDefinition(), param_name):
                p = limit_int_param(p, param_name, MIN_MODEL_EPOCHS, MAX_MODEL_EPOCHS)

        return dataset, p

    return func


@pytest.fixture
def helper_tune(tune_objective: Callable):
    def func(
        dataset, plugin, evaluation_case, metric, direction, sampler, pruner, override_hp_space, compute_baseline_score
    ):
        # Initialize and tune the tuner.OptunaTuner.
        hp_tuner = tuner.OptunaTuner(
            study_name="my_study",
            direction=direction,
            study_sampler=sampler,
            study_pruner=pruner,
        )
        return hp_tuner.tune(
            estimator=plugin,
            dataset=dataset,
            evaluation_callback=functools.partial(
                tune_objective,
                evaluation_case=evaluation_case,
                metric=metric,
            ),
            override_hp_space=override_hp_space,
            compute_baseline_score=compute_baseline_score,
            optimize_kwargs=OPTIMIZE_KWARGS,
        )

    return func


@pytest.fixture
def helper_asserts():
    def func(plugin, scores, params, compute_baseline_score: bool, pruner_enabled: bool):
        # Do the necessary asserts for the tests.
        hp_names = sorted(list(get_grid_by_sampling(plugin, n=1).keys()))
        assert len(scores) == len(params)
        if not pruner_enabled:
            assert len(params) == OPTIMIZE_KWARGS["n_trials"] + int(compute_baseline_score)
        else:
            assert len(params) > int(compute_baseline_score)
        if compute_baseline_score:
            assert params[0] == dict()  # Baseline score.
        for param in params[int(compute_baseline_score) :]:
            assert sorted(list(param.keys())) == hp_names

    return func


@pytest.fixture
def helper_test_optuna_tuner(
    helper_initialize: Callable,
    helper_tune: Callable,
    helper_asserts: Callable,
):
    def func(
        data: str,
        plugin: str,
        sampler: Any,
        pruner: Any,
        evaluation_case: str,
        metric: str,
        direction: OptimDirection,
        compute_baseline_score: bool,
    ):
        # A general test function for testing `tuner.OptunaTuner`.

        dataset, p = helper_initialize(
            data=data,
            plugin=plugin,
        )

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=SEED)

        if sampler == "GridSampler":
            sampler = optuna.samplers.GridSampler(search_space=get_grid_by_sampling(p, n=GRID_SAMPLER_GRID_N))

        scores, params = helper_tune(
            dataset=dataset,
            plugin=p,
            evaluation_case=evaluation_case,
            metric=metric,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            override_hp_space=None,
            compute_baseline_score=compute_baseline_score,
        )
        helper_asserts(
            plugin=p,
            compute_baseline_score=compute_baseline_score,
            scores=scores,
            params=params,
            pruner_enabled=pruner is not None,
        )

    return func


@pytest.fixture
def helper_test_optuna_tuner_pipeline_selector(
    get_dataset: Callable,
    helper_tune: Callable,
    helper_asserts: Callable,
):
    def func(
        data: str,
        pipeline_selector,
        sampler: Any,
        pruner: Any,
        evaluation_case: str,
        metric: str,
        direction: OptimDirection,
    ):
        # A general test function for testing `tuner.OptunaTuner` - pipeline selector case.

        dataset = get_dataset(data)

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=SEED)

        if sampler == "GridSampler":
            sampler = optuna.samplers.GridSampler(
                search_space=get_grid_by_sampling(pipeline_selector, n=GRID_SAMPLER_GRID_N)
            )

        scores, params = helper_tune(
            dataset=dataset,
            plugin=pipeline_selector,
            evaluation_case=evaluation_case,
            metric=metric,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            override_hp_space=None,
            compute_baseline_score=False,
        )
        helper_asserts(
            plugin=pipeline_selector,
            compute_baseline_score=False,
            scores=scores,
            params=params,
            pruner_enabled=pruner is not None,
        )

    return func


# TestOptunaTuner helper functions (end) ---


class TestOptunaTuner:
    # Evaluation case: prediction.one_off.classification.

    @pytest.mark.parametrize("compute_baseline_score", [True, False])
    def test_compute_baseline_score(
        self,
        compute_baseline_score: bool,
        helper_test_optuna_tuner: Callable,
    ):
        # Check compute_baseline_score = True, False cases.
        # helper_asserts() helper has appropriate asserts for each of these two cases.
        helper_test_optuna_tuner(
            data="sine_data_small",
            plugin="prediction.one_off.classification.nn_classifier",
            sampler=None,
            pruner=None,
            evaluation_case="prediction.one_off.classification",
            metric="aucroc",
            direction="maximize",
            compute_baseline_score=compute_baseline_score,
        )

    def test_override_hp_space(
        self,
        helper_initialize: Callable,
        helper_tune: Callable,
    ):
        # Can only be sampled as:
        # {'n_temporal_layers_hidden': 1, 'n_iter': 2}
        override = [
            IntegerParams(name="n_temporal_layers_hidden", low=1, high=1),
            IntegerParams(name="n_iter", low=2, high=2),
        ]

        dataset, p = helper_initialize(
            data="sine_data_small",
            plugin="prediction.one_off.classification.nn_classifier",
        )

        sampler = optuna.samplers.TPESampler(seed=SEED)

        scores, params = helper_tune(
            dataset=dataset,
            plugin=p,
            evaluation_case="prediction.one_off.classification",
            metric="aucroc",
            direction="maximize",
            sampler=sampler,
            pruner=None,
            override_hp_space=override,
            compute_baseline_score=False,
        )

        assert len(params) == len(scores) == OPTIMIZE_KWARGS["n_trials"]
        # Check that indeed our override params are being used. These can only be sampled as:
        # {'n_temporal_layers_hidden': 1, 'n_iter': 2}
        assert params[0] == params[1] == {"n_temporal_layers_hidden": 1, "n_iter": 2}

    def test_tune_no_optimize_kwargs(
        self,
        helper_initialize: Callable,
        tune_objective: Callable,
        monkeypatch,
    ):
        dataset, p = helper_initialize(
            data="sine_data_small",
            plugin="prediction.one_off.classification.nn_classifier",
        )

        sampler = optuna.samplers.TPESampler(seed=SEED)

        hp_tuner = tuner.OptunaTuner(
            study_name="my_study",
            direction="maximize",
            study_sampler=sampler,
            study_pruner=None,
        )
        monkeypatch.setattr(hp_tuner.study, "optimize", Mock(return_value=([], [])))

        hp_tuner.tune(
            estimator=p,
            dataset=dataset,
            evaluation_callback=functools.partial(
                tune_objective,
                evaluation_case="prediction.one_off.classification",
                metric="accuracy",
            ),
            optimize_kwargs=None,  # This.
        )

    def test_tune_empty_hp_space(
        self,
        helper_initialize: Callable,
        tune_objective: Callable,
        monkeypatch,
    ):
        dataset, p = helper_initialize(
            data="sine_data_small",
            plugin="prediction.one_off.classification.nn_classifier",
        )

        sampler = optuna.samplers.TPESampler(seed=SEED)

        hp_tuner = tuner.OptunaTuner(
            study_name="my_study",
            direction="maximize",
            study_sampler=sampler,
            study_pruner=None,
        )
        monkeypatch.setattr(p, "hyperparameter_space", Mock(return_value=[]))  # This.

        scores, params = hp_tuner.tune(
            estimator=p,
            dataset=dataset,
            evaluation_callback=functools.partial(
                tune_objective,
                evaluation_case="prediction.one_off.classification",
                metric="accuracy",
            ),
            optimize_kwargs=None,
            compute_baseline_score=False,
        )

        assert scores == params == []

    class TestPredictionOneOffClassification:
        @pytest.mark.parametrize("data", SETTINGS["prediction.one_off.classification"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["prediction.one_off.classification"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("sampler", SETTINGS["prediction.one_off.classification"]["TEST_SAMPLER_SET"])
        def test_tune_vary_samplers(self, data: str, plugin: str, sampler: Any, helper_test_optuna_tuner: Callable):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=sampler,
                pruner=None,
                evaluation_case="prediction.one_off.classification",
                metric="aucroc",
                direction="maximize",
                compute_baseline_score=False,
            )

        @pytest.mark.parametrize("data", SETTINGS["prediction.one_off.classification"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["prediction.one_off.classification"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("pruner", TEST_PRUNER_SET)
        def test_tune_vary_pruners(self, data: str, plugin: str, pruner: Any, helper_test_optuna_tuner: Callable):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=None,
                pruner=pruner,
                evaluation_case="prediction.one_off.classification",
                metric="aucroc",
                direction="maximize",
                compute_baseline_score=False,
            )

        @pytest.mark.parametrize("data", SETTINGS["prediction.one_off.classification"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["prediction.one_off.classification"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("metric,direction", SETTINGS["prediction.one_off.classification"]["METRICS"])
        def test_tune_vary_metrics(
            self,
            data: str,
            plugin: str,
            metric: str,
            direction: OptimDirection,
            helper_test_optuna_tuner: Callable,
        ):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=None,
                pruner=None,
                evaluation_case="prediction.one_off.classification",
                metric=metric,
                direction=direction,
                compute_baseline_score=False,
            )

    # Evaluation case: prediction.one_off.regression.

    class TestPredictionOneOffRegression:
        @pytest.mark.parametrize("data", SETTINGS["prediction.one_off.regression"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["prediction.one_off.regression"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("sampler", SETTINGS["prediction.one_off.regression"]["TEST_SAMPLER_SET"])
        def test_tune_vary_samplers(
            self,
            data: str,
            plugin: str,
            sampler: Any,
            helper_test_optuna_tuner: Callable,
        ):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=sampler,
                pruner=None,
                evaluation_case="prediction.one_off.regression",
                metric="mse",
                direction="minimize",
                compute_baseline_score=False,
            )

        @pytest.mark.parametrize("data", SETTINGS["prediction.one_off.regression"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["prediction.one_off.regression"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("pruner", TEST_PRUNER_SET)
        def test_tune_vary_pruners(self, data: str, plugin: str, pruner: Any, helper_test_optuna_tuner: Callable):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=None,
                pruner=pruner,
                evaluation_case="prediction.one_off.regression",
                metric="mse",
                direction="minimize",
                compute_baseline_score=False,
            )

        @pytest.mark.parametrize("data", SETTINGS["prediction.one_off.regression"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["prediction.one_off.regression"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("metric,direction", SETTINGS["prediction.one_off.regression"]["METRICS"])
        def test_tune_vary_metrics(
            self, data: str, plugin: str, metric: str, direction: OptimDirection, helper_test_optuna_tuner: Callable
        ):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=None,
                pruner=None,
                evaluation_case="prediction.one_off.regression",
                metric=metric,
                direction=direction,
                compute_baseline_score=False,
            )

    # Evaluation case: time_to_event.

    @pytest.mark.filterwarnings("ignore")  # Multiple warnings re. convergence etc.
    class TestTimeToEvent:
        @pytest.mark.parametrize("data", SETTINGS["time_to_event"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["time_to_event"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("sampler", SETTINGS["time_to_event"]["TEST_SAMPLER_SET"])
        def test_tune_vary_samplers(
            self,
            data: str,
            plugin: str,
            sampler: Any,
            helper_test_optuna_tuner: Callable,
        ):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=sampler,
                pruner=None,
                evaluation_case="time_to_event",
                metric="c_index",
                direction="maximize",
                compute_baseline_score=False,
            )

        @pytest.mark.filterwarnings("ignore")  # Multiple warnings re. convergence etc.
        @pytest.mark.parametrize("data", SETTINGS["time_to_event"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["time_to_event"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("pruner", TEST_PRUNER_SET)
        def test_tune_vary_pruners(self, data: str, plugin: str, pruner: Any, helper_test_optuna_tuner: Callable):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=None,
                pruner=pruner,
                evaluation_case="time_to_event",
                metric="c_index",
                direction="maximize",
                compute_baseline_score=False,
            )

        @pytest.mark.filterwarnings("ignore")  # Multiple warnings re. convergence etc.
        @pytest.mark.parametrize("data", SETTINGS["time_to_event"]["TEST_ON_DATASETS"])
        @pytest.mark.parametrize("plugin", SETTINGS["time_to_event"]["TEST_WITH_PLUGINS"])
        @pytest.mark.parametrize("metric,direction", SETTINGS["time_to_event"]["METRICS"])
        def test_tune_vary_metrics(
            self, data: str, plugin: str, metric: str, direction: OptimDirection, helper_test_optuna_tuner: Callable
        ):
            helper_test_optuna_tuner(
                data=data,
                plugin=plugin,
                sampler=None,
                pruner=None,
                evaluation_case="time_to_event",
                metric=metric,
                direction=direction,
                compute_baseline_score=False,
            )

    # Test PipelineSelector case.

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore")  # In some HyperImpute cases.
    def test_pipeline_selector_case(self, helper_test_optuna_tuner_pipeline_selector: Callable):
        from tempor.automl.pipeline_selector import PipelineSelector

        ps = PipelineSelector(task_type="prediction.one_off.classification", predictor="nn_classifier")

        data = "sine_data_full"

        helper_test_optuna_tuner_pipeline_selector(
            data=data,
            pipeline_selector=ps,
            sampler=None,
            pruner=None,
            evaluation_case="prediction.one_off.classification",
            metric="aucroc",
            direction="maximize",
        )
