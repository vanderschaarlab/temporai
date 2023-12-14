# mypy: ignore-errors

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data import Dataset
from ..data.constants import (
    DEFAULT_PADDING_INDICATOR,
    T_NumericDtype_AsTuple,
    T_TSIndexClass,
    T_TSIndexClass_AsTuple,
    T_TSIndexDtype,
    T_TSIndexDtype_AsTuple,
)
from ..utils.common import python_type_from_np_pd_dtype
from ..utils.dev import raise_not_implemented


class HorizonOpts(Enum):
    N_STEP_AHEAD = auto()
    TIME_INDEX = auto()
    # Other ideas: N_STEP_FORECAST = auto()


class Horizon(ABC):
    horizon_type: HorizonOpts

    def __init__(self, horizon_type: HorizonOpts) -> None:
        self.horizon_type = horizon_type


@dataclass(init=False, repr=True)  # For repr only.
class NStepAheadHorizon(Horizon):
    n_step: int

    def __init__(self, n_step: int) -> None:
        super().__init__(horizon_type=HorizonOpts.N_STEP_AHEAD)
        if n_step <= 0:
            raise ValueError("N step ahead horizon must be > 0.")
        self.n_step = n_step


TimeIndexSequence = Sequence[T_TSIndexClass]


# TODO: Unit-test.
# TODO: This is currently rough, and will change.
@dataclass(init=False, repr=True)  # For repr only.
class TimeIndexHorizon(Horizon):
    time_index_sequence: TimeIndexSequence  # TODO: Perhaps also allow for just T_TSIndexClass.

    def __init__(self, time_index_sequence: TimeIndexSequence) -> None:
        super().__init__(horizon_type=HorizonOpts.TIME_INDEX)
        for ti in time_index_sequence:
            if not isinstance(ti, T_TSIndexClass_AsTuple):
                raise ValueError(
                    f"Time index horizon `time_index` must be one of the following types: {T_TSIndexClass_AsTuple}"
                )
            if not issubclass(python_type_from_np_pd_dtype(ti.dtype), T_TSIndexDtype_AsTuple):  # type: ignore
                raise ValueError(f"Time index horizon `time_index` dtype must be one of: {T_TSIndexDtype_AsTuple}")
        self.time_index_sequence = time_index_sequence

    @classmethod
    def future_horizon_from_dataset(
        cls, data: Dataset, forecast_n_future_steps: int, time_delta: T_TSIndexDtype = 1
    ) -> "TimeIndexHorizon":
        targets = data.temporal_targets
        if targets is None:
            raise ValueError("Temporal targets must be set but was None")
        if not isinstance(time_delta, T_NumericDtype_AsTuple):
            raise_not_implemented(
                feature=f"{TimeIndexHorizon.__name__} from Dataset initialization when `time_delta` "
                f"is not one of: {T_NumericDtype_AsTuple}, was {type(time_delta)}"
            )
        targets_index_dtype = python_type_from_np_pd_dtype(targets.sample_index.dtype)  # type: ignore
        if not issubclass(targets_index_dtype, T_NumericDtype_AsTuple):
            raise_not_implemented(
                feature=f"{TimeIndexHorizon.__name__} from Dataset initialization when temporal targets index "
                f"is not one of: {T_NumericDtype_AsTuple}, was {targets_index_dtype}"
            )
        if not isinstance(time_delta, targets_index_dtype):
            raise ValueError(
                f"`time_delta` type ({type(time_delta)}) did not match the "
                f"temporal targets index dtype ({targets_index_dtype})"
            )
        indices = []
        for ts in targets:
            start = list(ts.time_index)[-1] + time_delta
            if TYPE_CHECKING:
                assert isinstance(time_delta, (int, float))
                assert isinstance(start, (int, float))
            seq = [start + time_delta * x for x in range(forecast_n_future_steps)]
            indices.append(pd.Index(seq))
        return cls(time_index_sequence=indices)

    def to_numpy_time_series(self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None):
        n_samples = len(self.time_index_sequence)
        if max_len is None:
            n_timesteps = max(len(ti) for ti in self.time_index_sequence)
        else:
            n_timesteps = max_len
        array = np.full(
            shape=(n_samples, n_timesteps, 1),
            fill_value=padding_indicator,
        )
        for idx, ti in enumerate(self.time_index_sequence):
            array[idx, : len(ti), 0] = np.asarray(ti.values)
        return array

    def to_torch_time_series(
        self,
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
        max_len: Optional[int] = None,
        **torch_tensor_kwargs,
    ):
        return torch.tensor(self.to_numpy_time_series(padding_indicator, max_len), **torch_tensor_kwargs)
