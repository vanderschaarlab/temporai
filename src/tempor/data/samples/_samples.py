from typing import Any, Dict, Optional

import pandas as pd

import tempor.data._settings as settings
import tempor.data._supports_container as sc
import tempor.data._types as types

from . import impl, interface


class Samples(interface.SamplesInterface, sc.SupportsContainer):
    # For any reused functionality in child {Static,TimeSeries,Event}Samples.

    def __init__(self, data: types.DataContainer, container_flavor: Optional[types.ContainerFlavor]) -> None:
        self._data = data
        if container_flavor is not None:
            self._container_flavor = container_flavor
        else:
            self._container_flavor = settings.DEFAULT_CONTAINER_FLAVOR[(self.data_category, type(self._data))]
        self._key = (type(self._data), self._container_flavor)
        sc.SupportsContainer.__init__(self)
        # For completeness only, not required:
        interface.SamplesInterface.__init__(self, self._data, self._container_flavor)

    @property
    def container_flavor(self) -> types.ContainerFlavor:
        return self.container_flavor

    @property
    def data(self):
        return self.dispatch_to_implementation(self._key).data

    def __str__(self) -> str:
        return f"{self.__class__.__name__}:\n{self.data}"


class StaticSamples(
    Samples,
    interface.StaticSamplesInterface,
    sc.StaticSupportsContainer[impl.StaticSamplesImplementation],
):
    def __init__(self, data: types.DataContainer, container_flavor: Optional[types.ContainerFlavor] = None) -> None:
        Samples.__init__(self, data, container_flavor)
        sc.StaticSupportsContainer.__init__(self)
        # For completeness only, not required:
        interface.StaticSamplesInterface.__init__(self, self._data, self._container_flavor)

    def as_data_frame(self):
        return self.dispatch_to_implementation(self._key).as_data_frame()

    def as_array(self):
        return self.dispatch_to_implementation(self._key).as_array()

    def _register_implementations(self) -> Dict[sc.DataContainerDef, impl.StaticSamplesImplementation]:
        df_sample_x_feature_def = (pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_X_FEATURE)
        df_sample_x_feature_impl = impl.df_samples.StaticSamples(
            data=self._data, container_flavor=self._container_flavor  # type: ignore
        )
        return {
            df_sample_x_feature_def: df_sample_x_feature_impl,
        }


class TimeSeriesSamples(
    Samples,
    interface.TimeSeriesSamplesInterface,
    sc.TimeSeriesSupportsContainer[impl.TimeSeriesSamplesImplementation],
):
    def __init__(self, data: types.DataContainer, container_flavor: Optional[types.ContainerFlavor] = None) -> None:
        Samples.__init__(self, data, container_flavor)
        sc.TimeSeriesSupportsContainer.__init__(self)
        # For completeness only, not required:
        interface.TimeSeriesSamplesInterface.__init__(self, self._data, self._container_flavor)

    def as_data_frame(self):
        return self.dispatch_to_implementation(self._key).as_data_frame()

    def as_array(self, padding_indicator: Any):
        return self.dispatch_to_implementation(self._key).as_array(padding_indicator)

    def _register_implementations(self) -> Dict[sc.DataContainerDef, impl.TimeSeriesSamplesImplementation]:
        df_sample_timestep_x_feature_def = (pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE)
        df_sample_timestep_x_feature_impl = impl.df_samples.TimeSeriesSamples(
            data=self._data, container_flavor=self._container_flavor  # type: ignore
        )
        return {
            df_sample_timestep_x_feature_def: df_sample_timestep_x_feature_impl,
        }


class EventSamples(
    Samples,
    interface.EventSamplesInterface,
    sc.EventSupportsContainer[impl.EventSamplesImplementation],
):
    def __init__(self, data: types.DataContainer, container_flavor: Optional[types.ContainerFlavor] = None) -> None:
        Samples.__init__(self, data, container_flavor)
        sc.EventSupportsContainer.__init__(self)
        # For completeness only, not required:
        interface.EventSamplesInterface.__init__(self, self._data, self._container_flavor)

    def as_data_frame(self):
        return self.dispatch_to_implementation(self._key).as_data_frame()

    def as_array(self):
        return self.dispatch_to_implementation(self._key).as_array()

    def _register_implementations(self) -> Dict[sc.DataContainerDef, impl.EventSamplesImplementation]:
        df_sample_x_time_event_def = (pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_TIME_X_EVENT)
        df_sample_x_time_event_impl = impl.df_samples.EventSamples(
            data=self._data, container_flavor=self._container_flavor  # type: ignore
        )
        return {
            df_sample_x_time_event_def: df_sample_x_time_event_impl,
        }
