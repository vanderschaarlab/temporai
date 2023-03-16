from typing import Any, List

from hyperimpute.plugins.imputers import Imputers as StaticImputers
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples, TimeSeriesSamples
from tempor.plugins.core._params import CategoricalParams, Params
from tempor.plugins.preprocessing.imputation import BaseImputer


@plugins.register_plugin(name="static_imputation", category="preprocessing.imputation")
class StaticOnlyImputer(BaseImputer):
    def __init__(
        self, static_imputer: str = "ice", temporal_imputer: str = "ice", random_state: int = 0, **params
    ) -> None:  # pylint: disable=useless-super-delegation
        """Impute the time-series using a static imputer.

        Args:
            static_imputer (str, optional):
                Which imputer to use for the static data (if any). Defaults to ``"ice"``.
            temporal_imputer (str, optional):
                Which imputer to use for the temporal data (if any). Defaults to ``"ice"``.
            random_state (int, optional):
                Random seed. Defaults to ``0``.
        """
        super().__init__(**params)
        self.static_imputer = StaticImputers().get(static_imputer, random_state=random_state)
        self.temporal_imputer = StaticImputers().get(temporal_imputer, random_state=random_state)
        self.random_state = random_state

    def _fit(self, data: dataset.Dataset, *args, **kwargs) -> Self:
        if data.static is not None:
            self.static_imputer.fit(data.static.dataframe())

        self.temporal_imputer.fit(data.time_series.dataframe())

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
        ts_data = data.time_series.dataframe()
        imputed_ts_data = self.temporal_imputer.transform(ts_data)
        imputed_ts_data.columns = ts_data.columns
        imputed_ts_data.index = ts_data.index

        data.time_series = TimeSeriesSamples.from_dataframe(imputed_ts_data)
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # pragma: no cover
        return [
            CategoricalParams(name="static_imputer", choices=["mean", "ice", "missforest"]),
            CategoricalParams(name="temporal_imputer", choices=["mean", "ice", "missforest"]),
        ]
