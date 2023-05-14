from typing import Any, Dict, List, Optional, Tuple, Type

import optuna

from tempor.core import utils
from tempor.core.types import PredictiveTaskType
from tempor.plugins import plugin_loader
from tempor.plugins.core import BaseEstimator, BasePredictor
from tempor.plugins.core._params import CategoricalParams, Params
from tempor.plugins.pipeline import PipelineBase, pipeline
from tempor.plugins.preprocessing.imputation import BaseImputer
from tempor.plugins.preprocessing.scaling import BaseScaler

PREFIX_STATIC_IMPUTERS = "preprocessing.imputation.static"
PREFIX_STATIC_SCALERS = "preprocessing.scaling.static"
PREFIX_TEMPORAL_IMPUTERS = "preprocessing.imputation.temporal"
PREFIX_TEMPORAL_SCALERS = "preprocessing.scaling.temporal"

DEFAULT_STATIC_IMPUTERS = plugin_loader.list()["preprocessing"]["imputation"]["static"]
DEFAULT_STATIC_SCALERS = plugin_loader.list()["preprocessing"]["scaling"]["static"]
DEFAULT_TEMPORAL_IMPUTERS = plugin_loader.list()["preprocessing"]["imputation"]["temporal"]
DEFAULT_TEMPORAL_SCALERS = plugin_loader.list()["preprocessing"]["scaling"]["temporal"]


def get_fqn(prefix: str, name: str) -> str:
    """Return a full plugin name from category / task type prefix and plugin name, that is ``{prefix}.{name}``."""
    return f"{prefix}.{name}"


class PipelineSelector:
    def __init__(
        self,
        task_type: PredictiveTaskType,
        predictor: str,
        static_imputers: List[str] = DEFAULT_STATIC_IMPUTERS,
        static_scalers: List[str] = DEFAULT_STATIC_SCALERS,
        temporal_imputers: List[str] = DEFAULT_TEMPORAL_IMPUTERS,
        temporal_scalers: List[str] = DEFAULT_TEMPORAL_SCALERS,
    ) -> None:
        """A helper class for AutoML pipeline selection.

        Defines custom version of methods:
            - ``hyperparameter_space``,
            - ``sample_hyperparameters``.

        Adds methods to create a pipeline from sampled hyperparameters:
            - ``pipeline_class_from_hps``,
            - ``pipeline_from_hps``.

        Provides the tools to create candidate pipelines where the imputers / scalers are exposed as a categorical
        hyperparameter in the hyperparameter space. The hyperparameter space of these, and the final predictor step
        are also sampled.

        Args:
            task_type (PredictiveTaskType):
                The task type of the predictors.
            predictor (str):
                The predictor estimator to be used as the last stage of the pipelines.
            static_imputers (List[str], optional):
                A list of candidate static imputers. Defaults to `DEFAULT_STATIC_IMPUTERS`.
            static_scalers (List[str], optional):
                A list of candidate static scalers. Defaults to `DEFAULT_STATIC_SCALERS`.
            temporal_imputers (List[str], optional):
                A list of candidate temporal imputers. Defaults to `DEFAULT_TEMPORAL_IMPUTERS`.
            temporal_scalers (List[str], optional):
                A list of candidate temporal scalers. Defaults to `DEFAULT_TEMPORAL_SCALERS`.
        """
        self.task_type: PredictiveTaskType = task_type

        self.static_imputers: List[Type[BaseImputer]] = [
            plugin_loader.get_class(get_fqn(PREFIX_STATIC_IMPUTERS, p)) for p in static_imputers
        ]
        self.static_scalers: List[Type[BaseScaler]] = [
            plugin_loader.get_class(get_fqn(PREFIX_STATIC_SCALERS, p)) for p in static_scalers
        ]
        self.temporal_imputers: List[Type[BaseImputer]] = [
            plugin_loader.get_class(get_fqn(PREFIX_TEMPORAL_IMPUTERS, p)) for p in temporal_imputers
        ]
        self.temporal_scalers: List[Type[BaseScaler]] = [
            plugin_loader.get_class(get_fqn(PREFIX_TEMPORAL_SCALERS, p)) for p in temporal_scalers
        ]

        self.predictor: Type[BasePredictor] = plugin_loader.get_class(get_fqn(str(self.task_type), predictor))

    def _preproc_candidate_lists(self) -> List[List[Type[BaseEstimator]]]:
        list_ = [self.static_imputers, self.static_scalers, self.temporal_imputers, self.temporal_scalers]
        return list_  # type: ignore

    def _preproc_prefixes(self) -> List[str]:
        return [PREFIX_STATIC_IMPUTERS, PREFIX_STATIC_SCALERS, PREFIX_TEMPORAL_IMPUTERS, PREFIX_TEMPORAL_SCALERS]

    @staticmethod
    def format_hps_names(plugin: Any, hps: List[Params]) -> List[Params]:
        """Format hyperparameter space of a plugin by giving each `Param` name as
        ``[plugin.name](hyperparameter.name)``. Necessary for differentiating hyperparameters of different stages in
        the pipeline.

        Args:
            plugin (Any):
                Plugin (method).
            hps (List[Params]):
                Hyperparameter space definition.

        Returns:
            List[Params]: Updated hyperparameter space definition.
        """
        for hp in hps:
            hp.name = f"[{plugin.name}]({hp.name})"
        return hps

    @staticmethod
    def _generate_candidates_param(candidates: List[Type[BaseEstimator]]) -> CategoricalParams:
        """Creates a `CategoricalParams` for choosing between the ``candidates`` models (used for the preprocessing
        steps of the pipeline).

        Args:
            candidates (List[Type[BaseEstimator]]):
                List of candidate plugins (methods).

        Returns:
            CategoricalParams: The `CategoricalParams` with the candidate names as ``choices`` and\
                named ``"<candidates>(category)"`` where ``category`` is plugin category (task type).
        """
        if candidates:
            category = candidates[0].category
            return CategoricalParams(name=f"<candidates>({category})", choices=[p.name for p in candidates])
        else:  # pragma: no cover
            # Should not reach here in normal use.
            raise RuntimeError("Must pass a list of candidates with at least one item")

    def hyperparameter_space(
        self,
        *args,
        override: Optional[List[Params]] = None,  # NOTE: Hyperparameter override applies to predictor only.
        **kwargs,
    ) -> List[Params]:
        """The customized ``hyperparameter_space`` implementation. The hyperparameter space is built up of the
        hyperparameters of each pipeline steps (the preprocessors, and the final predictive step). ``CategoricalParams``
        for choosing which preprocessor to use are also added (if at least one ``static_imputer``, ... are provided).
        Parameter names are customized to be able to differentiate pipeline stages (see ``format_hps_names``).

        Args:
            override (Optional[List[Params]], optional):
                Hyperparameter space override for the final predictive step of the pipeline. Defaults to `None`.

        Returns:
            List[Params]: Returned hyperparameter space.
        """
        hps: List[Params] = []

        for plugin_cls_list in self._preproc_candidate_lists():
            if plugin_cls_list:
                hps.append(self._generate_candidates_param(plugin_cls_list))
            for plugin in plugin_cls_list:
                hps.extend(self.format_hps_names(plugin, plugin.hyperparameter_space(*args, **kwargs)))

        if override is not None:
            hps.extend(self.format_hps_names(self.predictor, override))
        else:
            hps.extend(self.format_hps_names(self.predictor, self.predictor.hyperparameter_space(*args, **kwargs)))

        return hps

    def sample_hyperparameters(
        self,
        *args: Any,
        override: Optional[List[Params]] = None,  # NOTE: Hyperparameter override applies to predictor only.
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """The customized ``sample_hyperparameters`` implementation. Uses the customized ``hyperparameter_space``.

        Args:
            override (Optional[List[Params]], optional):
                Hyperparameter space override for the final predictive step of the pipeline. Defaults to `None`.

        Returns:
            Dict[str, Any]: Sampled hyperparameters dictionary.
        """
        # NOTE: The inner workings of this method need to stay roughly consistent with
        # `BaseEstimator.sample_hyperparameters`.

        # Pop the `trial` argument for optuna if such is found (if not, `trial` will be `None`).
        trial, args, kwargs = utils.get_from_args_or_kwargs(
            args, kwargs, argument_name="trial", argument_type=optuna.Trial, position_if_args=0
        )

        param_space = self.hyperparameter_space(*args, override=override, **kwargs)

        results = dict()

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results

    @staticmethod
    def _get_relevant_hps(plugin_name: str, hps: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dictionary of sampled hyperparameters for a particular pipeline step (the step for ``plugin_name``),
        from the dictionary of hyperparameters for the whole pipeline (``hps``). The customized hyperparameter names
        (see ``format_hps_names``) are turned back to normal hyperparameter names.

        Args:
            plugin_name (str):
                The plugin name corresponding to one of the stages in the pipeline.
            hps (Dict[str, Any]):
                The sampled hyperparameters for the entire pipeline.

        Returns:
            Dict[str, Any]: Returned hyperparameters for ``plugin_name``.
        """
        return {k.split("(")[-1][:-1]: v for k, v in hps.items() if f"[{plugin_name}]" in k}

    def pipeline_class_from_hps(self, hps: Dict[str, Any]) -> Tuple[Type[PipelineBase], Dict[str, Dict]]:
        """Return a pipeline class from the sampled hyperparameters ``hps``.

        Args:
            hps (Dict[str, Any]): The sampled hyperparameters for the pipeline.

        Returns:
            Tuple[Type[PipelineBase], Dict[str, Dict]]:
                ``(pipeline_cls, pipeline_params)``, the pipeline class and the params to initialize the pipeline with.
        """
        pipeline_def: List[str] = []
        pipeline_init_params: Dict[str, Dict] = dict()

        for candidates_list, prefix in zip(self._preproc_candidate_lists(), self._preproc_prefixes()):
            if candidates_list:
                selected = hps[f"<candidates>({prefix})"]
                selected_hps = self._get_relevant_hps(selected, hps)
                pipeline_def.append(get_fqn(prefix, selected))
                pipeline_init_params[selected] = selected_hps

        pipeline_def.append(self.predictor.fqn())
        predictor_hps = self._get_relevant_hps(self.predictor.name, hps)
        pipeline_init_params[self.predictor.name] = predictor_hps

        return pipeline(pipeline_def), pipeline_init_params

    def pipeline_from_hps(self, hps: Dict[str, Any]) -> PipelineBase:
        """Return a pipeline instance from the sampled hyperparameters ``hps``.

        Args:
            hps (Dict[str, Any]): The sampled hyperparameters for the pipeline.

        Returns:
            PipelineBase: The pipeline.
        """
        PipelineCls, pipeline_init_params = self.pipeline_class_from_hps(hps)

        return PipelineCls(pipeline_init_params)
