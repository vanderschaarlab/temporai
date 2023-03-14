"""Settings for TemporAI data handling."""

from typing import ClassVar, List

import pydantic

from . import data_typing


@pydantic.dataclasses.dataclass(frozen=True)
class DataSettings:
    """Dataclass containing TemporAI data settings, such as configuration for data validation."""

    static_value_dtypes: ClassVar[List[data_typing.Dtype]] = [bool, int, float, "category"]
    time_series_value_dtypes: ClassVar[List[data_typing.Dtype]] = [bool, int, float, "category"]
    event_value_dtypes: ClassVar[List[data_typing.Dtype]] = [bool]

    sample_index_dtypes: ClassVar[List[data_typing.Dtype]] = [int, str]
    time_index_dtypes: ClassVar[List[data_typing.Dtype]] = ["datetime", int, float]
    feature_index_dtypes: ClassVar[List[data_typing.Dtype]] = [str]

    sample_index_unique: ClassVar[bool] = True
    sample_timestep_index_unique: ClassVar[bool] = True
    sample_index_nullable: ClassVar[bool] = False
    time_index_nullable: ClassVar[bool] = False
    feature_index_nullable: ClassVar[bool] = False

    static_values_nullable: ClassVar[bool] = True
    time_series_values_nullable: ClassVar[bool] = True
    event_values_nullable: ClassVar[bool] = False

    sample_index_name: str = "sample_idx"
    time_index_name: str = "time_idx"

    default_padding_indicator: float = 999.0


DATA_SETTINGS = DataSettings()
"""TemporAI data settings."""
