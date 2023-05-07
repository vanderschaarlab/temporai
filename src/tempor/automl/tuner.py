"""Module containing the interface for, and the implemented hyperparameter tuners."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import optuna
from pydantic import validate_arguments
from typing_extensions import Protocol, runtime_checkable

from tempor.data.dataset import PredictiveDataset
from tempor.log import logger
from tempor.plugins.core._base_predictor import BasePredictor

from ._types import OptimDirection


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
        estimator: Type[BasePredictor],
        dataset: PredictiveDataset,
        evaluation_callback: EvaluationCallback,
        direction: OptimDirection,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Base hyperparameter tuner from which tuner implementations should derive. Defines the initializer and the
        `tune` method.

        Args:
            study_name (str):
                Study name.
            estimator (Type[BasePredictor]):
                Estimator class whose hyperparameters will be tuned.
            dataset (PredictiveDataset):
                Dataset to use.
            evaluation_callback (EvaluationCallback):
                Evaluation callback which will take in the estimator class, hyperparameters, and return a score.
            direction (OptimDirection):
                Optimization direction (`"minimize"` or `"maximize"`).
        """
        self.study_name = study_name
        self.estimator = estimator
        self.dataset = dataset
        self.evaluation_callback = evaluation_callback
        self.direction = direction

    @abc.abstractmethod
    def tune(self, compute_baseline_score: bool = True) -> Tuple[List[float], List[Dict]]:
        """Run the hyperparameter tuner and return scores and chosen hyperparameters.

        Args:
            compute_baseline_score (bool):
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
# TODO: Handle Pipeline.
# TODO: Handle ensembles.
# TODO: Early stopping pruner.
class OptunaTuner(BaseTuner):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        estimator: Type[BasePredictor],
        dataset: PredictiveDataset,
        evaluation_callback: EvaluationCallback,
        direction: OptimDirection,
        *,
        sampler: optuna.samplers.BaseSampler,
        study_storage: Optional[Union[str, optuna.storages.BaseStorage]] = None,
        study_pruner: Optional[optuna.pruners.BasePruner] = None,
        study_load_if_exists: bool = False,
        optimize_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Hyper parameter tuning (optimization) helper for an `optuna.study.Study` using any
        `optuna.sampler.BaseSampler`.

        Args:
            study_name (str):
                Study name.
            estimator (Type[BasePredictor]):
                Estimator class whose hyperparameters will be tuned.
            dataset (PredictiveDataset):
                Dataset to use.
            evaluation_callback (EvaluationCallback):
                Evaluation callback which will take in the estimator class, hyperparameters, and return a score.
            direction (OptimDirection):
                Optimization direction (`"minimize"` or `"maximize"`).
            sampler (optuna.samplers.BaseSampler):
                An `optuna` sampler (passed to `optuna.create_study`).
            study_storage (Optional[Union[str, optuna.storages.BaseStorage]], optional):
                An `optuna` storage object (passed to `optuna.create_study`). Defaults to `None`.
            study_pruner (Optional[optuna.pruners.BasePruner], optional):
                An `optuna` pruner (passed to `optuna.create_study`). Defaults to `None`.
            study_load_if_exists (bool, optional):
                The `load_if_exists` parameter (passed to `optuna.create_study`). Defaults to `False`.
            optimize_kwargs (Optional[Dict[str, Any]], optional):
                Keyword arguments to pass to ``study.optimize``. Defaults to `None`.
        """
        super().__init__(
            study_name=study_name,
            estimator=estimator,
            dataset=dataset,
            evaluation_callback=evaluation_callback,
            direction=direction,
        )

        self.sampler = sampler
        self.study_storage = study_storage
        self.study_pruner = study_pruner
        self.study_load_if_exists = study_load_if_exists
        if optimize_kwargs is None:
            optimize_kwargs = dict()
        self.optimize_kwargs = optimize_kwargs

        self.study = self._create_study()

    def _create_study(self) -> optuna.Study:
        study = optuna.create_study(
            storage=self.study_storage,
            sampler=self.sampler,
            pruner=self.study_pruner,
            study_name=self.study_name,
            direction=self.direction,
            load_if_exists=self.study_load_if_exists,
            directions=None,  # Do not support multi-objective.
        )
        return study

    def tune(self, compute_baseline_score: bool = True) -> Tuple[List[float], List[Dict]]:
        scores = []
        params: List[Dict[str, Any]] = []

        if compute_baseline_score:
            baseline_score = self.evaluation_callback(self.estimator, self.dataset)
            scores.append(baseline_score)
            params.append(dict())
            logger.info(f"Baseline score for {self.estimator.name}: {baseline_score}")
        else:
            logger.info("Baseline score computation skipped")

        if len(self.estimator.hyperparameter_space()) == 0:
            return scores, params

        def objective(trial: optuna.Trial) -> float:
            hps = self.estimator.sample_hyperparameters(trial)
            logger.info(f"Hyperparameters sampled from {self.estimator.__name__}:\n{hps}")
            score = self.evaluation_callback(self.estimator, self.dataset, **hps)
            return score

        self.study.optimize(objective, **self.optimize_kwargs)

        for trial_idx, trial_info in enumerate(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])):
            score_trial = trial_info.values[0]
            params_trial = trial_info.params
            logger.trace(f"Got trial {trial_idx}.")
            logger.trace(f"Trial score: {score_trial}.")
            logger.trace(f"Trial params:\n{params_trial}.")
            scores.append(score_trial)
            params.append(params_trial)

        return scores, params
