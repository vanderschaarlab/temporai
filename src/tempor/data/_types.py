import enum
from typing import Dict, Literal, Optional, Union

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


SamplesAttributes = Literal["Xt", "Xs", "Xe", "Yt", "Ys", "Ye", "At", "As", "Ae"]

ContainerFlavorSpec = Dict[SamplesAttributes, Optional[ContainerFlavor]]
