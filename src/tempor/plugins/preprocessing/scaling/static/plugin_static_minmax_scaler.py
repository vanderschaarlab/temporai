import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples
from tempor.plugins.preprocessing.scaling._base import BaseScaler


@plugins.register_plugin(name="static_minmax_scaler", category="preprocessing.scaling.static")
class StaticMinMaxScaler(BaseScaler):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        """MinMax scaling for the static data.

        Transform the static features by scaling each feature to a given range.
        This estimator scales and translates each feature individually such that it is in the given range on the
        training set, e.g. between zero and one.

        Example:
            >>> from tempor.utils.dataloaders import SineDataLoader
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> dataset = SineDataLoader().load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.scaling.static.static_minmax_scaler")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            StaticMinMaxScaler(...)
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

    def _transform(self, data: dataset.Dataset, *args, **kwargs) -> dataset.Dataset:
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
