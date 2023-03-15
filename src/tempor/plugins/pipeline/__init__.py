# stdlib
from typing import Any, Dict, List, Tuple, Type

# third party
from tempor.data import dataset
from tempor.plugins import plugin_loader

from .generators import (
    _generate_constructor,
    _generate_fit,
    _generate_get_args,
    _generate_hyperparameter_space_for_layer_impl,
    _generate_hyperparameter_space_impl,
    _generate_name_impl,
    _generate_predict,
    _generate_predict_proba,
    _generate_sample_param_impl,
)


class PipelineMeta(type):
    def __new__(cls: Type, name: str, plugins: Tuple[Type, ...], dct: dict) -> Any:
        dct["__init__"] = _generate_constructor()
        dct["fit"] = _generate_fit()
        dct["predict"] = _generate_predict()
        dct["predict_proba"] = _generate_predict_proba()
        dct["name"] = _generate_name_impl(plugins)
        dct["hyperparameter_space"] = _generate_hyperparameter_space_impl(plugins)
        dct["hyperparameter_space_for_layer"] = _generate_hyperparameter_space_for_layer_impl(plugins)
        dct["sample_params"] = _generate_sample_param_impl(plugins)
        dct["get_args"] = _generate_get_args()

        dct["plugin_types"] = list(plugins)

        return super().__new__(cls, name, tuple(), dct)

    @staticmethod
    def name(*args: Any) -> str:
        raise NotImplementedError("not implemented")

    @staticmethod
    def type(*args: Any) -> str:
        raise NotImplementedError("not implemented")

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> Dict:
        raise NotImplementedError("not implemented")

    @staticmethod
    def hyperparameter_space_for_layer(name: str, *args: Any, **kwargs: Any) -> Dict:
        raise NotImplementedError("not implemented")

    def sample_params(*args: Any, **kwargs: Any) -> Dict:
        raise NotImplementedError("not implemented")

    def get_args(*args: Any, **kwargs: Any) -> Dict:
        raise NotImplementedError("not implemented")

    def fit(self: Any, X: dataset.Dataset, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("not implemented")

    def predict(*args: Any, **kwargs: Any) -> dataset.Dataset:
        raise NotImplementedError("not implemented")

    def predict_proba(*args: Any, **kwargs: Any) -> dataset.Dataset:
        raise NotImplementedError("not implemented")


def PipelineGroup(names: List[str]) -> Tuple[Type, ...]:
    res = []

    for fqn in names:
        if "." not in fqn:
            raise RuntimeError("invalid fqn")

        res.append(plugin_loader.get_class(fqn))

    return tuple(res)


def Pipeline(plugins_str: List[str]) -> Any:
    plugins = PipelineGroup(plugins_str)

    name = "_".join(p.name for p in plugins)

    return PipelineMeta(name, plugins, {})
