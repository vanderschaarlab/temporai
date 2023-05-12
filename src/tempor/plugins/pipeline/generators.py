from typing import Any, Callable, Dict, Optional, Tuple, Type

from tempor.data import dataset


def _generate_pipeline_seq_impl(plugins: Tuple[Type, ...]) -> Callable:
    def pipeline_seq_impl(*args: Any) -> str:  # pylint: disable=unused-argument
        return "->".join(p.fqn() for p in plugins)

    return pipeline_seq_impl


def _generate_hyperparameter_space_impl(plugins: Tuple[Type, ...]) -> Callable:
    def hyperparameter_space_impl(*args: Any, **kwargs: Any) -> Dict:
        out = {}
        for p in plugins:
            out[p.name] = p.hyperparameter_space(*args, **kwargs)
        return out

    return hyperparameter_space_impl


def _generate_hyperparameter_space_for_step_impl(plugins: Tuple[Type, ...]) -> Callable:
    def hyperparameter_space_for_step_impl(step: str, *args: Any, **kwargs: Any) -> Dict:
        for p in plugins:
            if p.name == step:
                return p.hyperparameter_space(*args, **kwargs)
        raise ValueError(f"Invalid layer: {step}")

    return hyperparameter_space_for_step_impl


def _generate_sample_hyperparameters_impl(plugins: Tuple[Type, ...]) -> Callable:
    def sample_hyperparameters_impl(*args: Any, **kwargs: Any) -> Dict:  # pylint: disable=unused-argument
        sample: dict = {}
        for p in plugins:
            sample[p.name] = p.sample_hyperparameters()

        return sample

    return sample_hyperparameters_impl


def _generate_constructor() -> Callable:
    def _sanity_checks(plugins: Tuple[Type, ...]) -> None:
        if len(plugins) == 0:
            raise RuntimeError("Invalid empty pipeline.")

        for plugin in plugins[:-1]:
            if not hasattr(plugin, "transform"):
                raise RuntimeError(f"Invalid preprocessing plugin in the pipeline. {plugin}")

        if not hasattr(plugins[-1], "predict") and not hasattr(plugins[-1], "predict_counterfactuals"):
            raise RuntimeError(f"Invalid output plugin in the pipeline. {plugins[-1]}")

    def init_impl(self: Any, plugin_params: Optional[Dict[str, Dict]] = None) -> None:
        _sanity_checks(self.plugin_types)

        self.stages = []
        self.plugin_params = plugin_params if plugin_params is not None else dict()

        for plugin_type in self.plugin_types:
            plugin_args = {}
            if plugin_type.name in self.plugin_params:
                plugin_args = self.plugin_params[plugin_type.name]
            self.stages.append(plugin_type(**plugin_args))

    return init_impl


# def _generate_get_args() -> Callable:
#     def get_args_impl(self: Any) -> Dict:
#         return self.plugin_params
#     return get_args_impl


def _generate_fit() -> Callable:
    def fit_impl(self: Any, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> Any:
        local_X = data
        for stage in self.stages[:-1]:
            local_X = stage.fit_transform(local_X)

        self.stages[-1].fit(local_X, *args, **kwargs)

        return self

    return fit_impl


def _generate_predict() -> Callable:
    def predict_impl(self: Any, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:
        local_X = data
        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X)

        return self.stages[-1].predict(local_X, *args, **kwargs)

    return predict_impl


def _generate_predict_proba() -> Callable:
    def predict_proba_impl(self: Any, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:
        local_X = data
        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X, *args, **kwargs)

        return self.stages[-1].predict_proba(local_X)

    return predict_proba_impl


def _generate_predict_counterfactuals() -> Callable:
    def predict_counterfactuals_impl(self: Any, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:
        local_X = data
        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X, *args, **kwargs)

        return self.stages[-1].predict_counterfactuals(local_X, *args, **kwargs)

    return predict_counterfactuals_impl


__all__ = [
    "_generate_constructor",
    "_generate_fit",
    "_generate_hyperparameter_space_for_step_impl",
    "_generate_hyperparameter_space_impl",
    "_generate_pipeline_seq_impl",
    "_generate_predict_counterfactuals",
    "_generate_predict_proba",
    "_generate_predict",
    "_generate_sample_hyperparameters_impl",
]
