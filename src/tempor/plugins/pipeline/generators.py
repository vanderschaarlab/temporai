# stdlib
from typing import Any, Callable, Dict, Tuple, Type

# third party
from tempor.data import dataset


def _generate_name_impl(plugins: Tuple[Type, ...]) -> Callable:
    def name_impl(*args: Any) -> str:
        return "->".join(p.fqn() for p in plugins)

    return name_impl


def _generate_hyperparameter_space_impl(plugins: Tuple[Type, ...]) -> Callable:
    def hyperparameter_space_impl(*args: Any, **kwargs: Any) -> Dict:
        out = {}
        for p in plugins:
            out[p.name] = p.hyperparameter_space(*args, **kwargs)
        return out

    return hyperparameter_space_impl


def _generate_hyperparameter_space_for_layer_impl(plugins: Tuple[Type, ...]) -> Callable:
    def hyperparameter_space_for_layer_impl(layer: str, *args: Any, **kwargs: Any) -> Dict:
        for p in plugins:
            if p.name == layer:
                return p.hyperparameter_space(*args, **kwargs)
        raise ValueError(f"invalid layer {layer}")

    return hyperparameter_space_for_layer_impl


def _generate_sample_param_impl(plugins: Tuple[Type, ...]) -> Callable:
    def sample_param_impl(*args: Any, **kwargs: Any) -> Dict:
        sample: dict = {}
        for p in plugins:
            sample[p.name] = p.sample_hyperparameters()

        return sample

    return sample_param_impl


def _generate_constructor() -> Callable:
    def _sanity_checks(plugins: Tuple[Type, ...]) -> None:
        if len(plugins) == 0:
            raise RuntimeError("invalid empty pipeline.")

        for plugin in plugins[:-1]:
            if not hasattr(plugin, "transform"):
                raise RuntimeError(f"invalid preprocessing plugin in the pipeline. {plugin}")

        if not hasattr(plugins[-1], "predict"):
            raise RuntimeError(f"invalid output plugin in the pipeline. {plugins[-1]}")

    def init_impl(self: Any, args: dict = {}) -> None:
        _sanity_checks(self.plugin_types)

        self.stages = []
        self.args = args

        for plugin_type in self.plugin_types:
            plugin_args = {}
            if plugin_type.name in args:
                plugin_args = args[plugin_type.name]
            self.stages.append(plugin_type(**plugin_args))

    return init_impl


def _generate_get_args() -> Callable:
    def get_args_impl(self: Any) -> Dict:
        return self.args

    return get_args_impl


def _generate_fit() -> Callable:
    def fit_impl(self: Any, X: dataset.Dataset, *args: Any, **kwargs: Any) -> Any:
        local_X = X
        for stage in self.stages[:-1]:
            local_X = stage.fit_transform(local_X)

        self.stages[-1].fit(local_X, *args, **kwargs)

        return self

    return fit_impl


def _generate_predict() -> Callable:
    def predict_impl(self: Any, X: dataset.Dataset, *args: Any, **kwargs: Any) -> dataset.Dataset:
        local_X = X
        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X)

        return self.stages[-1].predict(local_X, *args, **kwargs)

    return predict_impl


def _generate_predict_proba() -> Callable:
    def predict_proba_impl(self: Any, X: dataset.Dataset, *args: Any, **kwargs: Any) -> dataset.Dataset:
        local_X = X
        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X, *args, **kwargs)

        return self.stages[-1].predict_proba(local_X)

    return predict_proba_impl


__all__ = [
    "_generate_name_impl",
    "_generate_hyperparameter_space_impl",
    "_generate_hyperparameter_space_for_layer_impl",
    "_generate_sample_param_impl",
    "_generate_constructor",
    "_generate_fit",
    "_generate_predict",
    "_generate_predict_proba",
    "_generate_get_args",
]
