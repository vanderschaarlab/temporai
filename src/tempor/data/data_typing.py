"""Types (and related code) for TemporAI data handling.
"""

import enum
from typing import Dict, List, Literal, Tuple, Type, Union

import numpy as np
import pandas as pd

DataContainer = Union[pd.DataFrame, np.ndarray]

Dtype = Union[Type, Literal["category", "datetime"]]
"""Type annotation to indicate dtypes. May be `Type`, e.g. `str`, `bool`, or one of the literals:
`"category"`, `"datetime"`
"""


class DataModality(enum.Enum):
    STATIC = enum.auto()
    TIME_SERIES = enum.auto()
    EVENT = enum.auto()


# Check these match `DATA_SETTINGS` in `data.settings``.
# - Sample index allowed dtypes:  (int, str).
# - Feature index allowed dtypes: (str,).
# - Time index allowed dtypes:    (pd.Timestamp, int, float).
# NOTE: The ordering of unions will affect pydantic parsing. Earlier types in the ordering will be tried first.
# See: https://docs.pydantic.dev/usage/types/#unions
SampleIndex = Union[
    List[int],
    List[str],
]
FeatureIndex = List[str]
TimeIndex = Union[
    List[float],
    List[int],
    List[pd.Timestamp],
]
SampleToTimeIndexDict = Union[
    Dict[int, TimeIndex],
    Dict[str, TimeIndex],
]
SampleTimeIndexTuples = List[
    Union[
        Tuple[int, float],
        Tuple[int, int],
        Tuple[int, pd.Timestamp],
        Tuple[str, float],
        Tuple[str, int],
        Tuple[str, pd.Timestamp],
    ]
]
TimeIndexList = List[TimeIndex]


class PredictiveTask(enum.Enum):
    ONE_OFF_PREDICTION = enum.auto()
    TEMPORAL_PREDICTION = enum.auto()
    TIME_TO_EVENT_ANALYSIS = enum.auto()
    ONE_OFF_TREATMENT_EFFECTS = enum.auto()
    TEMPORAL_TREATMENT_EFFECTS = enum.auto()
