from typing import Any, List

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.plugins.core._params import CategoricalParam, Params
from tempor.plugins.preprocessing.imputation import BaseImputer


@plugins.register_plugin(name="ffill", category="preprocessing.imputation")
class FFillImputer(BaseImputer):
    def __init__(
        self, static_imputer: str = "mean", random_state: int = 0, **params
    ) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)
        self.static_imputer = static_imputer
        self.random_state = random_state

    def _fit(self, data: dataset.Dataset, *args, **kwargs) -> "FFillImputer":
        return self

    def _transform(self, data: dataset.Dataset, *args, **kwargs) -> Any:
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # pragma: no cover
        return [CategoricalParam(name="static_imputer", choices=["mean", "ice", "missforest"])]
