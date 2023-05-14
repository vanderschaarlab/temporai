"""Module containing the interface for, and the implemented hyperparameter tuners."""

import abc
import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import optuna
from pydantic import validate_arguments
from typing_extensions import Protocol, runtime_checkable

from tempor.data.dataset import PredictiveDataset
from tempor.log import logger
from tempor.plugins.core._base_predictor import BasePredictor
from tempor.plugins.core._params import Params

from ._types import OptimDirection
from .pipeline_selector import PipelineSelector


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
        ...


class BaseTuner(abc.ABC):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        direction: OptimDirection,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Base hyperparameter tuner from which tuner implementations should derive. Defines the initializer and the
        `tune` method.

        Args:
            study_name (str):
                Study name.
            direction (OptimDirection):
                Optimization direction (`"minimize"` or `"maximize"`).
        """
        self.study_name = study_name
        self.direction = direction

    @abc.abstractmethod
    def tune(
        self,
        estimator: Union[Type[BasePredictor], PipelineSelector],
        dataset: PredictiveDataset,
        evaluation_callback: EvaluationCallback,
        override_hp_space: Optional[List[Params]] = None,
        compute_baseline_score: bool = True,
        **kwargs,
    ) -> Tuple[List[float], List[Dict]]:
        """Run the hyperparameter tuner and return scores and chosen hyperparameters.

        Args:
            estimator (Union[Type[BasePredictor], PipelineSelector]):
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

        Returns:
            Tuple[List[float], List[Dict]]:
                ``(scores, params)`` tuple, containing a list of scores for the tuning runs and a list of dictionaries\
                containing the parameters for each corresponding tuning run.
        """
        ...  # pylint: disable=unnecessary-ellipsis


# TODO: Handle other hyperparameter tuning frameworks, e.g. hyperband.

# TODO: Handle other storage types, e.g. redis.
# TODO: Possibly add a repeated parameter pruner.
# TODO: Handle ensembles.
class OptunaTuner(BaseTuner):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        direction: OptimDirection,
        *,
        study_sampler: optuna.samplers.BaseSampler,
        study_storage: Optional[Union[str, optuna.storages.BaseStorage]] = None,
        study_pruner: Optional[optuna.pruners.BasePruner] = None,
        study_load_if_exists: bool = False,
        **kwargs,
    ):
        """Hyper parameter tuning (optimization) helper for an `optuna.study.Study` using any
        `optuna.sampler.BaseSampler`.

        Args:
            study_name (str):
                Study name.
            direction (OptimDirection):
                Optimization direction (`"minimize"` or `"maximize"`).
            study_sampler (optuna.samplers.BaseSampler):
                An `optuna` sampler (passed to `optuna.create_study`).
            study_storage (Optional[Union[str, optuna.storages.BaseStorage]], optional):
                An `optuna` storage object (passed to `optuna.create_study`). Defaults to `None`.
            study_pruner (Optional[optuna.pruners.BasePruner], optional):
                An `optuna` pruner (passed to `optuna.create_study`). Defaults to `None`.
            study_load_if_exists (bool, optional):
                The `load_if_exists` parameter (passed to `optuna.create_study`). Defaults to `False`.
        """
        super().__init__(
            study_name=study_name,
            direction=direction,
        )

        self.sampler = study_sampler
        self.study_storage = study_storage
        self.study_pruner = study_pruner
        self.study_load_if_exists = study_load_if_exists

        self.create_study()

    def create_study(self) -> optuna.Study:
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
        estimator: Union[Type[BasePredictor], PipelineSelector],
        dataset: PredictiveDataset,
        evaluation_callback: EvaluationCallback,
        override_hp_space: Optional[List[Params]] = None,
        compute_baseline_score: bool = True,
        optimize_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[List[float], List[Dict]]:
        """Run the hyperparameter tuner and return scores and chosen hyperparameters.

        Args:
            estimator (Union[Type[BasePredictor], PipelineSelector]):
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

        Tuple[List[float], List[Dict]]:
                ``(scores, params)`` tuple, containing a list of scores for the tuning runs and a list of dictionaries\
                containing the parameters for each corresponding tuning run.
        """

        if optimize_kwargs is None:
            optimize_kwargs = dict()

        scores = []
        params: List[Dict[str, Any]] = []

        if compute_baseline_score and not isinstance(estimator, PipelineSelector):
            # NOTE: in case of PipelineSelector base case is not defined, so this is ignored.
            baseline_score = evaluation_callback(estimator, dataset)
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
                if TYPE_CHECKING:  # pragma: no cover
                    assert issubclass(pipe_cls, BasePredictor)  # nosec B101
                estimator_for_eval = pipe_cls
            else:
                estimator_for_eval = estimator
                name = estimator_for_eval.__name__

            logger.info(f"Hyperparameters sampled from {name}:\n{hps}")
            score = evaluation_callback(estimator_for_eval, dataset, **hps)

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
