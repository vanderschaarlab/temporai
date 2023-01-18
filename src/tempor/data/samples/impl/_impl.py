import tempor.data as dat
from tempor.data import validator as v
from tempor.data.samples import interface


class SamplesImplementation:
    # For any reused functionality in child {Static,TimeSeries,Event}SamplesImplementation.

    def __init__(self, data: dat.DataContainer, container_flavor: dat.ContainerFlavor) -> None:
        self._data = data
        self._container_flavor = container_flavor

    @property
    def data(self):
        return self._data


class StaticSamplesImplementation(SamplesImplementation, interface.StaticSamplesInterface):
    def __init__(self, data: dat.DataContainer, container_flavor: dat.ContainerFlavor) -> None:
        self._data = v.StaticDataValidator().validate(data=data, requirements=[], container_flavor=container_flavor)
        SamplesImplementation.__init__(self, data, container_flavor)
        interface.StaticSamplesInterface.__init__(self, data, container_flavor)  # For completeness.


class TimeSeriesSamplesImplementation(SamplesImplementation, interface.TimeSeriesSamplesInterface):
    def __init__(self, data: dat.DataContainer, container_flavor: dat.ContainerFlavor) -> None:
        self._data = v.TimeSeriesDataValidator().validate(data=data, requirements=[], container_flavor=container_flavor)
        SamplesImplementation.__init__(self, data, container_flavor)
        interface.TimeSeriesSamplesInterface.__init__(self, data, container_flavor)  # For completeness.


class EventSamplesImplementation(SamplesImplementation, interface.EventSamplesInterface):
    def __init__(self, data: dat.DataContainer, container_flavor: dat.ContainerFlavor) -> None:
        self._data = v.EventDataValidator().validate(data=data, requirements=[], container_flavor=container_flavor)
        SamplesImplementation.__init__(self, data, container_flavor)
        interface.EventSamplesInterface.__init__(self, data, container_flavor)  # For completeness.
