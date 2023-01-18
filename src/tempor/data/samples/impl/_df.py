from typing import TYPE_CHECKING, Any

import pandas as pd

import tempor.data as dat
from tempor.data import utils

from . import _impl as impl


class SamplesImplementationDF(impl.SamplesImplementation):
    # For any reused functionality in child {Static,TimeSeries,Event}SamplesImplementationDF.

    @property
    def data(self) -> pd.DataFrame:
        if TYPE_CHECKING:
            assert isinstance(self._data, pd.DataFrame)
        return self._data


class StaticSamples(SamplesImplementationDF, impl.StaticSamplesImplementation):
    def __init__(self, data: pd.DataFrame, container_flavor: dat.ContainerFlavor) -> None:
        SamplesImplementationDF.__init__(self, data, container_flavor)
        impl.StaticSamplesImplementation.__init__(self, data, container_flavor)

    def as_data_frame(self):
        return self.data

    def as_array(self):
        return self.data.to_numpy()


class TimeSeriesSamples(SamplesImplementationDF, impl.TimeSeriesSamplesImplementation):
    def __init__(self, data: pd.DataFrame, container_flavor: dat.ContainerFlavor) -> None:
        SamplesImplementationDF.__init__(self, data, container_flavor)
        impl.TimeSeriesSamplesImplementation.__init__(self, data, container_flavor)

    def as_data_frame(self):
        return self._data

    def as_array(self, padding_indicator: Any):
        return utils.multiindex_timeseries_df_to_array(
            df=self.data, padding_indicator=padding_indicator, max_timesteps=None
        )


class EventSamples(SamplesImplementationDF, impl.EventSamplesImplementation):
    def __init__(self, data: pd.DataFrame, container_flavor: dat.ContainerFlavor) -> None:
        SamplesImplementationDF.__init__(self, data, container_flavor)
        impl.EventSamplesImplementation.__init__(self, data, container_flavor)

    def as_data_frame(self):
        return self._data

    def as_array(self):
        return self.data.to_numpy()
