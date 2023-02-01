import tempor.data._types as types
from tempor.data.container import _validator as v
from tempor.data.samples import interface


class SamplesImplementation:
    # For any reused functionality in child {Static,TimeSeries,Event}SamplesImplementation.

    def __init__(self, data: types.DataContainer, container_flavor: types.ContainerFlavor) -> None:
        self._data = data
        self._container_flavor = container_flavor

    @property
    def data(self):
        return self._data


class StaticSamplesImplementation(SamplesImplementation, interface.StaticSamplesInterface):
    def __init__(self, data: types.DataContainer, container_flavor: types.ContainerFlavor) -> None:
        self._data = v.StaticDataValidator().validate(target=data, requirements=[], container_flavor=container_flavor)
        SamplesImplementation.__init__(self, data, container_flavor)
        interface.StaticSamplesInterface.__init__(self, data, container_flavor)  # For completeness.


class TimeSeriesSamplesImplementation(SamplesImplementation, interface.TimeSeriesSamplesInterface):
    def __init__(self, data: types.DataContainer, container_flavor: types.ContainerFlavor) -> None:
        self._data = v.TimeSeriesDataValidator().validate(
            target=data, requirements=[], container_flavor=container_flavor
        )
        SamplesImplementation.__init__(self, data, container_flavor)
        interface.TimeSeriesSamplesInterface.__init__(self, data, container_flavor)  # For completeness.


class EventSamplesImplementation(SamplesImplementation, interface.EventSamplesInterface):
    def __init__(self, data: types.DataContainer, container_flavor: types.ContainerFlavor) -> None:
        self._data = v.EventDataValidator().validate(target=data, requirements=[], container_flavor=container_flavor)
        SamplesImplementation.__init__(self, data, container_flavor)
        interface.EventSamplesInterface.__init__(self, data, container_flavor)  # For completeness.
