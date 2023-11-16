"""Module containing the interface for, and the implemented hyperparameter tuners."""

import abc
import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import optuna
import pydantic
from typing_extensions import Protocol, runtime_checkable

from tempor.core import pydantic_utils
from tempor.data.dataset import PredictiveDataset
from tempor.log import logger
from tempor.methods.core._base_predictor import BasePredictor
from tempor.methods.core.params import Params
from tempor.metrics import metric_typing

from ._types import AutoMLCompatibleEstimator
from .pipeline_selector import PipelineSelector

# TODO: Handle other hyperparameter tuning frameworks, e.g. hyperband.
# TODO: Explicitly handle other storage types, e.g. redis.
# TODO: Possibly add a repeated parameter pruner.
# TODO: Support ensembles.


@runtime_checkable
class EvaluationCallback(Protocol):
    """Evaluation callback callable.

    Should take in `Type[BasePredictor]` as first argument, then the dataset (`PredictiveDataset`),
    hyperparameters, returning a `float` score.

    Signature like:\
        ``(estimator: Type[BasePredictor], dataset: PredictiveDataset, *args: Any, **kwargs: Any) -> float``.
    """

    def __call__(
        self, estimator: Type[BasePredictor], dataset: PredictiveDataset, *args: Any, **kwargs: Any
    ) -> float:  # pragma: no cover
        """Evaluation callback call.

        Args:
            estimator (Type[BasePredictor]): Any predictor.
            dataset (PredictiveDataset): Any predictive dataset.
            *args (Any): Any additional arguments.
            **kwargs (Any): Any additional keyword arguments.

        Returns:
            float: Evaluation value/score.
        """
        ...  # pylint: disable=unnecessary-ellipsis


class BaseTuner(abc.ABC):
    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def __init__(  # pylint: disable=unused-argument
        self,
        study_name: str,
        direction: metric_typing.MetricDirection,
        **kwargs: Any,
    ):
        """Base hyperparameter tuner from which tuner implementations should derive. Defines the initializer and the
        `tune` method.

        Args:
            study_name (str):
                Study name.
            direction (metric_typing.MetricDirection):
                Optimization direction (`"minimize"` or `"maximize"`).
            **kwargs (Any):
                Currently unused.
        """
        self.study_name = study_name
        self.direction = direction

    @abc.abstractmethod
    def tune(
        self,
        estimator: AutoMLCompatibleEstimator,
        dataset: PredictiveDataset,
        evaluation_callback: EvaluationCallback,
        override_hp_space: Optional[List[Params]] = None,
        compute_baseline_score: bool = True,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Dict]]:  # pragma: no cover
        """Run the hyperparameter tuner and return scores and chosen hyperparameters.

        Args:
            estimator (AutoMLCompatibleEstimator):
                Estimator class, or `PipelineSelector`, whose hyperparameters will be tuned.
            dataset (PredictiveDataset):
                Dataset to use.
            evaluation_callback (EvaluationCallback):
                Evaluation callback which will take in the estimator class, hyperparameters, and return a score.
            override_hp_space (Optional[List[Params]]):
                If this is not `None`, hyperparameters will be sampled from this list, rather than from those defined
                in the ``hyperparameter_space`` method of the estimator. Defaults to `None`.
            compute_baseline_score (bool, optional):
                If `True`, a trial will be run with default parameters (hyperparameters passed to ``__init__`` as an
                empty dictionary). This will be returned as the zeroth item in ``scores`` and ``params``. If `False`,
                this will be skipped. Defaults to `True`.
            **kwargs (Any):
                Currently unused.

        Returns:
            Tuple[List[float], List[Dict]]:
                ``(scores, params)`` tuple, containing a list of scores for the tuning runs and a list of dictionaries\
                containing the parameters for each corresponding tuning run.
        """
        ...  # pylint: disable=unnecessary-ellipsis


class OptunaTuner(BaseTuner):
    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        direction: metric_typing.MetricDirection,
        *,
        study_sampler: optuna.samplers.BaseSampler,
        study_storage: Optional[Union[str, optuna.storages.BaseStorage]] = None,
        study_pruner: Optional[optuna.pruners.BasePruner] = None,
        study_load_if_exists: bool = False,
        **kwargs: Any,
    ):
        """Hyper parameter tuning (optimization) helper for an `optuna.study.Study` using any
        `optuna.sampler.BaseSampler`.

        Args:
            study_name (str):
                Study name.
            direction (metric_typing.MetricDirection):
                Optimization direction (`"minimize"` or `"maximize"`).
            study_sampler (optuna.samplers.BaseSampler):
                An `optuna` sampler (passed to `optuna.create_study`).
            study_storage (Optional[Union[str, optuna.storages.BaseStorage]], optional):
                An `optuna` storage object (passed to `optuna.create_study`). Defaults to `None`.
            study_pruner (Optional[optuna.pruners.BasePruner], optional):
                An `optuna` pruner (passed to `optuna.create_study`). Defaults to `None`.
            study_load_if_exists (bool, optional):
                The `load_if_exists` parameter (passed to `optuna.create_study`). Defaults to `False`.
            **kwargs (Any):
                Currently unused.
        """
        super().__init__(
            study_name=study_name,
            direction=direction,
            **kwargs,
        )

        self.sampler = study_sampler
        self.study_storage = study_storage
        self.study_pruner = study_pruner
        self.study_load_if_exists = study_load_if_exists

        self.create_study()

    def create_study(self) -> optuna.Study:
        """Create an `optuna.Study` to be used for tuning. Sets the ``self.study`` attribute.

        Returns:
            optuna.Study: The created study.
        """
        self.study = optuna.create_study(
            storage=self.study_storage,
            sampler=self.sampler,
            pruner=self.study_pruner,
            study_name=self.study_name,
            direction=self.direction,
            load_if_exists=self.study_load_if_exists,
            directions=None,  # Do not support multi-objective.
        )
        return self.study

    def tune(
        self,
        estimator: AutoMLCompatibleEstimator,
        dataset: PredictiveDataset,
        evaluation_callback: EvaluationCallback,
        override_hp_space: Optional[List[Params]] = None,
        compute_baseline_score: bool = True,
        optimize_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Dict]]:
        """Run the hyperparameter tuner and return scores and chosen hyperparameters.

        Args:
            estimator (AutoMLCompatibleEstimator):
                Estimator class, or `PipelineSelector`, whose hyperparameters will be tuned.
            dataset (PredictiveDataset):
                Dataset to use.
            evaluation_callback (EvaluationCallback):
                Evaluation callback which will take in the estimator class, hyperparameters, and return a score.
            override_hp_space (Optional[List[Params]]):
                If this is not `None`, hyperparameters will be sampled from this list, rather than from those defined
                in the ``hyperparameter_space`` method of the estimator. Defaults to `None`.
            compute_baseline_score (bool, optional):
                If `True`, a trial will be run with default parameters (hyperparameters passed to ``__init__`` as an
                empty dictionary). This will be returned as the zeroth item in ``scores`` and ``params``. If `False`,
                this will be skipped. Defaults to `True`.
            optimize_kwargs (Optional[Dict[str, Any]], optional):
                Keyword arguments to pass to ``study.optimize``. Defaults to `None`.
            **kwargs (Any):
                Currently unused.

        Returns:
            Tuple[List[float], List[Dict]]:
                ``(scores, params)`` tuple, containing a list of scores for the tuning runs and a list of dictionaries\
                containing the parameters for each corresponding tuning run.
        """

        if optimize_kwargs is None:
            optimize_kwargs = dict()

        scores = []
        params: List[Dict[str, Any]] = []

        # TODO: Handle dataset passing through transformation such that there is no need to copy.

        if compute_baseline_score and not isinstance(estimator, PipelineSelector):
            # NOTE: in case of PipelineSelector base case is not defined, so this is ignored.
            baseline_score = evaluation_callback(estimator, copy.deepcopy(dataset))
            scores.append(baseline_score)
            params.append(dict())
            logger.info(f"Baseline score for {estimator.name}: {baseline_score}")
        else:
            logger.info("Baseline score computation skipped")

        if len(estimator.hyperparameter_space()) == 0:
            return scores, params

        def objective(trial: optuna.Trial) -> float:
            # Ensure the override variable doesn't get mutated unintentionally by copying.
            override_copy = copy.deepcopy(override_hp_space)
            hps = estimator.sample_hyperparameters(trial, override=override_copy)

            estimator_for_eval: Type[BasePredictor]
            if isinstance(estimator, PipelineSelector):
                pipe_cls, pipe_hp_dict = estimator.pipeline_class_from_hps(hps)
                hps = dict(plugin_params=pipe_hp_dict)
                name = pipe_cls.pipeline_seq()
                estimator_for_eval = cast(Type[BasePredictor], pipe_cls)
            else:
                estimator_for_eval = estimator
                name = estimator_for_eval.__name__

            logger.info(f"Hyperparameters sampled from {name}:\n{hps}")
            score = evaluation_callback(estimator_for_eval, copy.deepcopy(dataset), **hps)

            return score

        self.study.optimize(objective, **optimize_kwargs)

        for trial_idx, trial_info in enumerate(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])):
            score_trial = trial_info.values[0]
            params_trial = trial_info.params
            logger.trace(f"Got trial {trial_idx}.")
            logger.trace(f"Trial score: {score_trial}.")
            logger.trace(f"Trial params:\n{params_trial}.")
            scores.append(score_trial)
            params.append(params_trial)

        return scores, params
