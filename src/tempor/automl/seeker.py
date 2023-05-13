import copy
import functools
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pydantic
from typing_extensions import Literal, Type, get_args

from tempor.benchmarks import evaluation
from tempor.core.types import PredictiveTaskType
from tempor.data import data_typing
from tempor.data.dataset import PredictiveDataset, TimeToEventAnalysisDataset
from tempor.log import logger
from tempor.plugins import plugin_loader
from tempor.plugins.core import BasePredictor
from tempor.plugins.core._params import Params

from ._types import OptimDirection
from .tuner import BaseTuner, OptunaTuner

TunerType = Literal[
    "bayesian",
    "random",
    "cmaes",
    "qmc",
    "grid",
]
"""Hyperparameter tuner to use.

Available options:
  - `"bayesian"`: Use a tuner based on `optuna.samplers.TPESampler`.
  - `"random"`: Use a tuner based on `optuna.samplers.RandomSampler`.
  - `"cmaes"`: Use a tuner based on `optuna.samplers.CmaEsSampler`.
  - `"qmc"`: Use a tuner based on `optuna.samplers.QMCSampler`.
  - `"grid"`: Use a tuner based on `optuna.samplers.GridSampler`.
"""

SupportedMetric = Union[
    evaluation.ClassifierSupportedMetric,
    evaluation.RegressionSupportedMetric,
    evaluation.TimeToEventSupportedMetric,
]

TUNER_OPTUNA_SAMPLER_MAP: Dict[TunerType, Any] = {
    "bayesian": optuna.samplers.TPESampler,
    "random": optuna.samplers.RandomSampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "qmc": optuna.samplers.QMCSampler,
    "grid": optuna.samplers.GridSampler,
}

METRIC_DIRECTION_MAP: Dict[SupportedMetric, OptimDirection] = {
    # ClassifierSupportedMetric:
    "aucroc": "maximize",
    "aucprc": "maximize",
    "accuracy": "maximize",
    "f1_score_micro": "maximize",
    "f1_score_macro": "maximize",
    "f1_score_weighted": "maximize",
    "kappa": "maximize",
    "kappa_quadratic": "maximize",
    "precision_micro": "maximize",
    "precision_macro": "maximize",
    "precision_weighted": "maximize",
    "recall_micro": "maximize",
    "recall_macro": "maximize",
    "recall_weighted": "maximize",
    "mcc": "maximize",
    # RegressionSupportedMetric:
    "mse": "minimize",
    "mae": "minimize",
    "r2": "maximize",
    # TimeToEventMetricCallable:
    "c_index": "maximize",
    "brier_score": "minimize",
}

# TODO: Pipeline with imputer and scaler.
# TODO: Docstring.

# TODO: Parallelism support (per-estimator).


def evaluation_callback_dispatch(
    estimator: Type[BasePredictor],
    dataset: PredictiveDataset,
    task_type: PredictiveTaskType,
    metric: str,
    n_cv_folds: int,
    random_state: int,
    horizon: Optional[data_typing.TimeIndex],
    raise_exceptions: bool,
    *args,
    **kwargs,
) -> float:
    model = estimator(*args, **kwargs)
    # TODO: Handle missing cases.
    if task_type == "prediction.one_off.classification":
        metrics = evaluation.evaluate_prediction_oneoff_classifier(
            model,
            dataset,
            n_splits=n_cv_folds,
            random_state=random_state,
            raise_exceptions=raise_exceptions,
        )
    elif task_type == "prediction.one_off.regression":
        metrics = evaluation.evaluate_prediction_oneoff_regressor(
            model,
            dataset,
            n_splits=n_cv_folds,
            random_state=random_state,
            raise_exceptions=raise_exceptions,
        )
    elif task_type == "prediction.temporal.classification":  # pragma: no cover
        raise NotImplementedError
    elif task_type == "prediction.temporal.regression":  # pragma: no cover
        raise NotImplementedError
    elif task_type == "time_to_event":
        if not isinstance(dataset, TimeToEventAnalysisDataset):
            raise ValueError(
                f"`dataset` must be of type {TimeToEventAnalysisDataset.__name__} for the '{task_type}' task"
            )
        if horizon is None:
            raise ValueError(f"`horizon` must not be None for the '{task_type}' task")
        metrics = evaluation.evaluate_time_to_event(
            model,
            dataset,
            horizons=horizon,
            n_splits=n_cv_folds,
            random_state=random_state,
            raise_exceptions=raise_exceptions,
        )
    elif task_type == "treatments.one_off.classification":  # pragma: no cover
        raise NotImplementedError
    elif task_type == "treatments.one_off.regression":  # pragma: no cover
        raise NotImplementedError
    elif task_type == "treatments.temporal.classification":  # pragma: no cover
        raise NotImplementedError
    elif task_type == "treatments.temporal.regression":  # pragma: no cover
        raise NotImplementedError
    else:  # pragma: no cover
        raise ValueError(f"Unknown evaluation case: {task_type}")
    return metrics.loc[metric, "mean"]  # pyright: ignore


class SimpleSeeker:
    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        task_type: PredictiveTaskType,
        estimator_names: List[str],
        metric: SupportedMetric,
        dataset: PredictiveDataset,
        *,
        return_top_k: int = 3,
        num_cv_folds: int = 5,
        num_iter: int = 100,
        tuner_patience: int = 5,
        tuner_type: TunerType = "bayesian",
        timeout: int = 360,
        random_state: int = 0,
        override_hp_space: Optional[Dict[str, List[Params]]] = None,
        horizon: Optional[data_typing.TimeIndex] = None,
        compute_baseline_score: bool = False,
        grid: Optional[Dict[str, Dict[str, Any]]] = None,
        custom_tuner: Optional[BaseTuner] = None,
        raise_exceptions: bool = True,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        if override_hp_space is None:
            override_hp_space = dict()

        self.study_name = study_name
        self.task_type: PredictiveTaskType = task_type
        self.estimator_names = estimator_names
        self.metric: SupportedMetric = metric
        self.dataset = dataset
        self.return_top_k = return_top_k
        self.num_cv_folds = num_cv_folds
        self.num_iter = num_iter
        self.tuner_patience = tuner_patience
        self.tuner_type: TunerType = tuner_type
        self.timeout = timeout
        self.random_state = random_state
        self.horizon = horizon
        self.compute_baseline_score = compute_baseline_score
        self.grid = grid
        self.raise_exceptions = raise_exceptions

        # Validate "grid" case:
        if self.tuner_type == "grid":
            if self.grid is None:
                raise ValueError("Must provide `grid` argument when using the 'grid' search method")
            if self.estimator_names != list(self.grid.keys()):
                raise ValueError(
                    f"`grid` keys ({list(self.grid.keys())}) did not match " f"estimator names ({self.estimator_names})"
                )

        # Validate and set override_hp_space:
        for key in override_hp_space.keys():
            if key not in self.estimator_names:
                raise ValueError(
                    f"Estimator name '{key}' in `override_hp_space` was found that did not "
                    f"correspond to any of the estimators provided: {self.estimator_names}"
                )
        self.override_hp_space = override_hp_space

        # Validate custom tuner.
        if isinstance(custom_tuner, OptunaTuner):
            if isinstance(custom_tuner.sampler, optuna.samplers.GridSampler):
                raise ValueError("Passing a custom tuner with `optuna.samplers.GridSampler` is not supported")
        self.custom_tuner = custom_tuner

        self.direction: OptimDirection = METRIC_DIRECTION_MAP[self.metric]
        self.estimators: List[Type[BasePredictor]] = []
        self.tuners: List[BaseTuner] = []

        self._set_up_tuners()

    def _set_up_tuners(self):
        logger.info(f"Setting up estimators and tuners for study {self.study_name}.")
        for estimator_name in self.estimator_names:
            # Set up estimator.
            estimator_fqn = f"{self.task_type}.{estimator_name}"
            EstimatorCls = plugin_loader.get_class(estimator_fqn)
            self.estimators.append(EstimatorCls)

            # Set up tuner.
            if self.custom_tuner is None:
                # Case: Default tuners based on tuner_type etc.
                # Pruner:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
                    pruner = optuna.pruners.PatientPruner(None, self.tuner_patience)
                # Sampler:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
                    SamplerCls = TUNER_OPTUNA_SAMPLER_MAP[self.tuner_type]
                    if self.tuner_type != "grid":
                        sampler = SamplerCls(seed=self.random_state)
                    else:
                        if TYPE_CHECKING:  # pragma: no cover
                            assert self.grid is not None  # nosec B101
                        sampler = SamplerCls(seed=self.random_state, search_space=self.grid[estimator_name])
                # Instantiate:
                if self.tuner_type in TUNER_OPTUNA_SAMPLER_MAP:
                    tuner = OptunaTuner(
                        study_name=f"{self.study_name}_{EstimatorCls.name}",
                        direction=self.direction,
                        study_sampler=sampler,
                        study_storage=None,
                        study_pruner=pruner,
                        study_load_if_exists=False,
                    )
                    self.tuners.append(tuner)
                else:  # pragma: no cover
                    # Should not get here as there is pydantic validation.
                    raise ValueError(f"Unsupported tuner type: {self.tuner_type}. Available: {get_args(TunerType)}.")
            else:
                # Case: Custom tuner.
                # Copy the tuner for each estimator.
                if TYPE_CHECKING:  # pragma: no cover
                    assert isinstance(self.custom_tuner, OptunaTuner)  # nosec B101
                study_name = f"{self.study_name}_{EstimatorCls.name}"
                custom_tuner = copy.deepcopy(self.custom_tuner)
                custom_tuner.study_name = study_name  # Update the study name to be unique.
                custom_tuner.create_study()
                self.tuners.append(custom_tuner)

    def search(self) -> Tuple[List[BasePredictor], List[float]]:
        search_results: List[Tuple[List[float], List[Dict]]] = []
        for idx, (estimator_name, estimator_cls, tuner) in enumerate(
            zip(self.estimator_names, self.estimators, self.tuners)
        ):
            logger.info(f"Running  search for estimator '{estimator_cls.name}' {idx+1}/{len(self.estimators)}.")

            if estimator_name in self.override_hp_space:
                override = self.override_hp_space[estimator_name]
            else:
                override = None

            estimator_results = tuner.tune(
                estimator=estimator_cls,
                dataset=self.dataset,
                evaluation_callback=functools.partial(
                    evaluation_callback_dispatch,
                    task_type=self.task_type,
                    metric=self.metric,
                    n_cv_folds=self.num_cv_folds,
                    random_state=self.random_state,
                    horizon=self.horizon,
                    raise_exceptions=self.raise_exceptions,
                ),
                override_hp_space=override,
                compute_baseline_score=self.compute_baseline_score,
                # NOTE: The below is OptunaTuner-only kwarg:
                optimize_kwargs=dict(n_trials=self.num_iter, timeout=self.timeout),
            )
            search_results.append(estimator_results)

        all_estimators = []
        all_scores = []
        all_hps = []
        for idx, (scores, hps) in enumerate(search_results):
            best_idx = np.argmin(scores) if self.direction == "minimize" else np.argmax(scores)
            all_scores.append(scores[best_idx])
            all_hps.append(hps[best_idx])
            all_estimators.append(self.estimators[idx])
            logger.info(f"Evaluation for {self.estimators[idx].name} scores:\n{scores}.")
        logger.info(f"All estimator definitions searched:\n{self.estimator_names}")
        logger.info(f"Best scores for each estimator searched:\n{all_scores}")
        logger.info(f"Best hyperparameters for each estimator searched:\n{all_hps}")

        all_scores_np = np.array(all_scores)
        top_k = min(self.return_top_k, len(all_scores))
        unique_sorted = np.sort(np.unique(all_scores_np.ravel()))
        if self.direction == "minimize":
            unique_sorted = unique_sorted[::-1]
        top_k_scores = unique_sorted[-top_k:]

        result: List[Any] = []
        result_scores: List[float] = []
        for score in reversed(top_k_scores):
            result_scores.append(score)
            idx = np.argwhere(all_scores_np == score)[0][0]
            logger.info(f"Selected score {score} for {all_estimators[idx].name} with hyperparameters:\n{all_hps[idx]}")
            estimator_top_k = all_estimators[idx](**all_hps[idx])
            result.append(estimator_top_k)

        return result, result_scores
