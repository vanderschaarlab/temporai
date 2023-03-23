from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples
from tempor.plugins.preprocessing.scaling import BaseScaler


@plugins.register_plugin(name="static_standard_scaler", category="preprocessing.scaling")
class StaticStandardScaler(BaseScaler):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        """Standard scaling for the static data.
        Standardize the static features by removing the mean and scaling to unit variance.

        Example:
            >>> from tempor.utils.dataloaders import SineDataLoader
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> dataset = SineDataLoader().load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.scaling.static_standard_scaler")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            StaticStandardScaler(...)
            >>>
            >>> # Scale:
            >>> scaled = model.transform(dataset)
        """

        super().__init__(**params)
        self.model = StandardScaler()

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
