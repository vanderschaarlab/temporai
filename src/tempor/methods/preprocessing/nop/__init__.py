from typing_extensions import Self

import tempor.methods.core as methods_core
from tempor.core import plugins
from tempor.data import dataset

plugins.register_plugin_category("preprocessing.nop", methods_core.BaseTransformer)


@plugins.register_plugin(name="nop_transformer", category="preprocessing.nop")
class NopTransformer(methods_core.BaseTransformer):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> Self:
        return self

    def _transform(self, data: dataset.BaseDataset, *args, **kwargs) -> dataset.BaseDataset:
        return data

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []


__all__ = [
    "NopTransformer",
]
