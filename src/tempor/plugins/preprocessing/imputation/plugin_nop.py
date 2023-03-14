from typing import Any

from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.plugins.preprocessing.imputation import BaseImputer


@plugins.register_plugin(name="nop_imputer", category="preprocessing.imputation")
class NopImputer(BaseImputer):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Self:
        return self

    def _transform(self, data: dataset.Dataset, *args, **kwargs) -> Any:
        return data

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []
