import abc
from typing import Tuple

import pandas as pd

import tempor.data.container._check_data_container_def as check
from tempor.core import supports_impl as si

from . import _types as types

DataContainerDef = Tuple[type, types.ContainerFlavor]


class SupportsContainer(si.SupportsImplementations[DataContainerDef, si.ImplementationT], abc.ABC):
    def check_data_container_supported_types(self, data_container: types.DataContainer):
        # Each derived class of `SupportsContainer` may choose to support only a subset of data container types.
        # Hence this pydantic validator added to check that.
        supports_container_defs: Tuple[DataContainerDef, ...] = self.supports_implementations_for
        supports_containers = tuple(x[0] for x in supports_container_defs)
        if not isinstance(data_container, supports_containers):
            raise TypeError(
                f"`data_container` must be one of the supported types {supports_containers} but was {type(data_container)}"
            )

    def __init__(self) -> None:
        for container_class, container_flavor in self.supports_implementations_for:
            check.CheckDataContainerDefinition(
                data_category=self.data_category, container_class=container_class, container_flavor=container_flavor
            )
        super().__init__()

    @property
    @abc.abstractmethod
    def data_category(self) -> types.DataCategory:  # pragma: no cover
        ...


class StaticSupportsContainer(SupportsContainer[si.ImplementationT]):
    @property
    def data_category(self) -> types.DataCategory:
        return types.DataCategory.STATIC

    @property
    def supports_implementations_for(self) -> Tuple[DataContainerDef, ...]:
        return ((pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_X_FEATURE),)


class TimeSeriesSupportsContainer(SupportsContainer[si.ImplementationT]):
    @property
    def data_category(self) -> types.DataCategory:
        return types.DataCategory.TIME_SERIES

    @property
    def supports_implementations_for(self) -> Tuple[DataContainerDef, ...]:
        return ((pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE),)


class EventSupportsContainer(SupportsContainer[si.ImplementationT]):
    @property
    def data_category(self) -> types.DataCategory:
        return types.DataCategory.EVENT

    @property
    def supports_implementations_for(self) -> Tuple[DataContainerDef, ...]:
        return ((pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_TIME_X_EVENT),)
