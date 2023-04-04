from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset

plugins.register_plugin_category("preprocessing.nop", plugins.BaseTransformer)


@plugins.register_plugin(name="nop_transformer", category="preprocessing.nop")
class NopTransformer(plugins.BaseTransformer):
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
