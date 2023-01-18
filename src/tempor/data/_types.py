import enum
from typing import Literal, Union

import numpy as np
import pandas as pd

DataContainer = Union[pd.DataFrame, np.ndarray]
Dtype = Union[type, Literal["category", "datetime"]]


class DataCategory(enum.Enum):
    STATIC = enum.auto()
    TIME_SERIES = enum.auto()
    EVENT = enum.auto()


class ContainerFlavor(enum.Enum):
    DF_SAMPLE_X_FEATURE = enum.auto()
    DF_SAMPLE_TIMESTEP_X_FEATURE = enum.auto()
    DF_SAMPLE_TIME_X_EVENT = enum.auto()
