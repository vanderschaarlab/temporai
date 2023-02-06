import tempor.plugins.core as plugins
from tempor.data.bundle._bundle import DataBundle as Dataset
from tempor.plugins.preprocessing.imputation import BaseImputer


@plugins.register_plugin(name="nop_imputer", category="preprocessing.imputation")
class NopImputer(BaseImputer):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def _fit(self, data: Dataset, *args, **kwargs) -> "NopImputer":
        return self

    def _transform(self, data: Dataset, *args, **kwargs) -> Dataset:
        return data
