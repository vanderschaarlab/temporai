from typing import Any

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.plugins.preprocessing.scaling import BaseScaler


@plugins.register_plugin(name="nop_scaler", category="preprocessing.scaling")
class NopScaler(BaseScaler):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def _fit(self, data: dataset.Dataset, *args, **kwargs) -> "NopScaler":
        return self

    def _transform(self, data: dataset.Dataset, *args, **kwargs) -> Any:
        return data
