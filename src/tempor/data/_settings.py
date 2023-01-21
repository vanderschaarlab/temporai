from typing import ClassVar, Dict, Set, Tuple, Type

import pandas as pd
import pydantic.dataclasses

from . import _types as types


@pydantic.dataclasses.dataclass(frozen=True)
class DataSettings:
    value_dtypes: ClassVar[Set[types.Dtype]] = {bool, int, float, "category"}
    sample_index_dtypes: ClassVar[Set[types.Dtype]] = {int, str}
    time_index_dtypes: ClassVar[Set[types.Dtype]] = {"datetime", int}
    feature_index_dtypes: ClassVar[Set[types.Dtype]] = {str}
    sample_index_unique: ClassVar[bool] = True
    sample_timestep_index_unique: ClassVar[bool] = True
    sample_index_nullable: ClassVar[bool] = False
    time_index_nullable: ClassVar[bool] = False
    feature_index_nullable: ClassVar[bool] = False
    values_nullable: ClassVar[bool] = True
    sample_index_name: str = "sample_idx"
    time_index_name: str = "time_idx"


DATA_SETTINGS = DataSettings()


DATA_CATEGORY_TO_CONTAINER_FLAVORS: Dict[types.DataCategory, Set[types.ContainerFlavor]] = {
    types.DataCategory.STATIC: {types.ContainerFlavor.DF_SAMPLE_X_FEATURE},
    types.DataCategory.TIME_SERIES: {types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE},
    types.DataCategory.EVENT: {types.ContainerFlavor.DF_SAMPLE_TIME_X_EVENT},
}


CONTAINER_CLASS_TO_CONTAINER_FLAVORS: Dict[Type[types.DataContainer], Set[types.ContainerFlavor]] = {
    pd.DataFrame: {
        types.ContainerFlavor.DF_SAMPLE_X_FEATURE,
        types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE,
        types.ContainerFlavor.DF_SAMPLE_TIME_X_EVENT,
    },
}


DEFAULT_CONTAINER_FLAVOR: Dict[Tuple[types.DataCategory, type], types.ContainerFlavor] = {
    (types.DataCategory.STATIC, pd.DataFrame): types.ContainerFlavor.DF_SAMPLE_X_FEATURE,
    (types.DataCategory.TIME_SERIES, pd.DataFrame): types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE,
    (types.DataCategory.EVENT, pd.DataFrame): types.ContainerFlavor.DF_SAMPLE_TIME_X_EVENT,
}
