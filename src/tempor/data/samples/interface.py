from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

import tempor.data as dat


class SamplesInterface(ABC):
    @abstractmethod
    def __init__(self, data: dat.DataContainer, container_flavor: Optional[dat.ContainerFlavor] = None) -> None:
        ...

    @property
    @abstractmethod
    def data(self) -> dat.DataContainer:
        ...

    @abstractmethod
    def as_data_frame(self) -> pd.DataFrame:
        ...

    def as_df(self) -> pd.DataFrame:  # Alias.
        return self.as_data_frame()


class StaticSamplesInterface(SamplesInterface):
    @abstractmethod
    def as_array(self) -> np.ndarray:
        ...


class TimeSeriesSamplesInterface(SamplesInterface):
    @abstractmethod
    def as_array(self, padding_indicator) -> np.ndarray:
        ...


class EventSamplesInterface(SamplesInterface):
    pass
