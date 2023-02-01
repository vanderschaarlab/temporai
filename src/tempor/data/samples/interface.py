import abc
from typing import Optional

import numpy as np
import pandas as pd

import tempor.data._types as types


class SamplesInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, data: types.DataContainer, container_flavor: Optional[types.ContainerFlavor] = None) -> None:
        ...

    @property
    @abc.abstractmethod
    def data(self) -> types.DataContainer:
        ...

    @abc.abstractmethod
    def as_data_frame(self) -> pd.DataFrame:
        ...

    def as_df(self) -> pd.DataFrame:  # Alias.
        return self.as_data_frame()


class StaticSamplesInterface(SamplesInterface):
    @abc.abstractmethod
    def as_array(self) -> np.ndarray:
        ...


class TimeSeriesSamplesInterface(SamplesInterface):
    @abc.abstractmethod
    def as_array(self, padding_indicator) -> np.ndarray:
        ...


class EventSamplesInterface(SamplesInterface):
    @abc.abstractmethod
    def as_array(self) -> np.ndarray:
        ...
