from typing import Any, List

from hyperimpute.plugins.imputers import Imputers as StaticImputers

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples, TimeSeriesSamples
from tempor.plugins.core._params import CategoricalParam, Params
from tempor.plugins.preprocessing.imputation import BaseImputer


@plugins.register_plugin(name="bfill", category="preprocessing.imputation")
class BFillImputer(BaseImputer):
    """
    Backward-first Time-Series Imputation

    Args:
        static_imputer: str
            Which imputer to use for the static data(if any)
        random_state: int
            Random seed
    """

    def __init__(
        self, static_imputer: str = "mean", random_state: int = 0, **params
    ) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)
        self.static_imputer = StaticImputers().get(static_imputer, random_state=random_state)
        self.random_state = random_state

    def _fit(self, data: dataset.Dataset, *args, **kwargs) -> "BFillImputer":
        if data.static is not None:
            self.static_imputer.fit(data.static.dataframe())

        return self

    def _transform(self, data: dataset.Dataset, *args, **kwargs) -> Any:
        # impute static data
        if data.static is not None:
            static_data = data.static.dataframe()
            imputed_static_data = self.static_imputer.transform(static_data)
            imputed_static_data.columns = static_data.columns
            imputed_static_data.index = static_data.index

            data.static = StaticSamples.from_dataframe(imputed_static_data)

        # impute temporal data
        sample_ts_index = data.time_series.sample_index()
        imputed_ts = data.time_series.dataframe()
        for idx in sample_ts_index:
            imputed_ts.loc[(idx,)].bfill(inplace=True)
            imputed_ts.loc[(idx,)].ffill(inplace=True)
            imputed_ts.loc[(idx,)].fillna(0, inplace=True)

        data.time_series = TimeSeriesSamples.from_dataframe(imputed_ts)
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # pragma: no cover
        return [CategoricalParam(name="static_imputer", choices=["mean", "ice", "missforest"])]
