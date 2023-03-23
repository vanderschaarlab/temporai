from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.data.samples import TimeSeriesSamples
from tempor.plugins.preprocessing.scaling import BaseScaler


@plugins.register_plugin(name="ts_minmax_scaler", category="preprocessing.scaling")
class TimeSeriesMinMaxScaler(BaseScaler):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        """MinMax scaling for the time-series data.
        Transform the temporal features by scaling each feature to a given range.
        This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.


        Example:
            >>> from tempor.utils.dataloaders import SineDataLoader
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> dataset = SineDataLoader().load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.scaling.ts_minmax_scaler")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            TimeSeriesMinMaxScaler(...)
            >>>
            >>> # Scale:
            >>> scaled = model.transform(dataset)
        """

        super().__init__(**params)
        self.model = MinMaxScaler()

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Self:
        self.model.fit(data.time_series.dataframe())
        return self

    def _transform(self, data: dataset.Dataset, *args, **kwargs) -> Any:
        temporal_data = data.time_series.dataframe()
        scaled = pd.DataFrame(self.model.transform(temporal_data))
        scaled.columns = temporal_data.columns
        scaled.index = temporal_data.index

        data.time_series = TimeSeriesSamples.from_dataframe(scaled)

        return data

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []
