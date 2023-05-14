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
    return f"{prefix}.{name}"


# TODO: Docstrings.
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

        self.predictor: Type[BasePredictor] = plugin_loader.get_class(get_fqn(self.task_type, predictor))

    def _preproc_candidate_lists(self) -> List[List[Type[BaseEstimator]]]:
        list_ = [self.static_imputers, self.static_scalers, self.temporal_imputers, self.temporal_scalers]
        return list_  # type: ignore

    def _preproc_prefixes(self) -> List[str]:
        return [PREFIX_STATIC_IMPUTERS, PREFIX_STATIC_SCALERS, PREFIX_TEMPORAL_IMPUTERS, PREFIX_TEMPORAL_SCALERS]

    @staticmethod
    def format_hps_names(plugin: Any, hps: List[Params]) -> List[Params]:
        for hp in hps:
            hp.name = f"[{plugin.name}]({hp.name})"
        return hps

    @staticmethod
    def _generate_candidates_param(candidates: List[Type[BaseEstimator]]) -> CategoricalParams:
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
        return {k.split("(")[-1][:-1]: v for k, v in hps.items() if f"[{plugin_name}]" in k}

    def pipeline_class_from_hps(self, hps: Dict[str, Any]) -> Tuple[Type[PipelineBase], Dict[str, Dict]]:
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
        PipelineCls, pipeline_init_params = self.pipeline_class_from_hps(hps)

        return PipelineCls(pipeline_init_params)
