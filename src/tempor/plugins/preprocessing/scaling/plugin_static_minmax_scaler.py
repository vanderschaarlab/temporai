from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples
from tempor.plugins.preprocessing.scaling import BaseScaler


@plugins.register_plugin(name="static_minmax_scaler", category="preprocessing.scaling")
class StaticMinMaxScaler(BaseScaler):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        """MinMax scaling for the static data.

        Example:
            >>> from tempor.utils.datasets.sine import SineDataloader
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> dataset = SineDataloader().load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.scaling.static_minmax_scaler")
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
        if data.static is None:
            return self

        self.model.fit(data.static.dataframe())
        return self

    def _transform(self, data: dataset.Dataset, *args, **kwargs) -> Any:
        if data.static is None:
            return data

        static_data = data.static.dataframe()
        scaled = pd.DataFrame(self.model.transform(static_data))
        scaled.columns = static_data.columns
        scaled.index = static_data.index

        data.static = StaticSamples.from_dataframe(scaled)

        return data

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []
