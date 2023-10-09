import dataclasses
from typing import Any, Dict

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing_extensions import Self

import tempor.methods.core as plugins
from tempor.data import dataset
from tempor.data.samples import TimeSeriesSamples
from tempor.methods.preprocessing.scaling._base import BaseScaler


@dataclasses.dataclass
class TimeSeriesStandardScalerParams:
    """Initialization parameters for :class:`TimeSeriesStandardScaler`."""

    with_mean: bool = True
    """If True, center the data before scaling. See `sklearn.preprocessing.StandardScaler`."""
    with_std: bool = True
    """If True, scale the data to unit variance. See `sklearn.preprocessing.StandardScaler`."""


@plugins.register_plugin(name="ts_standard_scaler", category="preprocessing.scaling.temporal")
class TimeSeriesStandardScaler(BaseScaler):
    ParamsDefinition = TimeSeriesStandardScalerParams
    params: TimeSeriesStandardScalerParams  # type: ignore

    def __init__(self, **params) -> None:
        """Standard scaling for the time-series data.

        Standardize the temporal features by removing the mean and scaling to unit variance. The time series data
        will be represented as a multi-index `(sample_idx, time_idx)` dataframe of features, and the scaling will be
        applied to this dataframe.

        Args:
            **params:
                Parameters and defaults as defined in :class:`TimeSeriesStandardScalerParams`.

        Example:
            >>> from tempor.utils.dataloaders import SineDataLoader
            >>> from tempor.methods import plugin_loader
            >>>
            >>> dataset = SineDataLoader().load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.scaling.temporal.ts_standard_scaler")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            TimeSeriesStandardScaler(...)
            >>>
            >>> # Scale:
            >>> scaled = model.transform(dataset)
        """
        super().__init__(**params)
        sklearn_params: Dict[str, Any] = dict(self.params)  # type: ignore
        self.model = StandardScaler(**sklearn_params)

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> Self:
        self.model.fit(data.time_series.dataframe())
        return self

    def _transform(self, data: dataset.BaseDataset, *args, **kwargs) -> dataset.BaseDataset:
        temporal_data = data.time_series.dataframe()
        scaled = pd.DataFrame(self.model.transform(temporal_data))
        scaled.columns = temporal_data.columns
        scaled.index = temporal_data.index

        data.time_series = TimeSeriesSamples.from_dataframe(scaled)

        return data

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []
