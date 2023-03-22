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

        Example:
            >>> from tempor.utils.datasets.sine import SineDataloader
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> dataset = SineDataloader().load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.scaling.ts_minmax_scaler")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
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
