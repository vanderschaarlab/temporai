from typing import Any, List

from typing_extensions import Self

import tempor.methods.core as methods_core
from tempor.core import plugins
from tempor.data import dataset
from tempor.methods.core.params import Params

plugins.register_plugin_category("preprocessing.nop", methods_core.BaseTransformer)


@plugins.register_plugin(name="nop_transformer", category="preprocessing.nop")
class NopTransformer(methods_core.BaseTransformer):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        return self

    def _transform(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> dataset.BaseDataset:
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # noqa: D102
        return []


__all__ = [
    "NopTransformer",
]
