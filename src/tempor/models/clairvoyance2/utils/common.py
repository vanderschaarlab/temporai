from typing import Any, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from dotmap import DotMap


def _extract_np_dtype_from_pd(_type: type) -> type:
    """Helper to handle `pandas` references to `numpy` dtypes, e.g. `dtype('int64')`.
    If `_type` is a `pandas` reference to a `numpy` dtype, will return the underlying `numpy` dtype.
    Otherwise will return `_type` as passed.
    """
    try:
        _type = _type.type  # type: ignore
    except AttributeError:
        pass
    return _type


# For the purposes of data type comparisons in this library,
# we assume the following numpy types be equivalent to Python types.
NP_EQUIVALENT_TYPES_MAP: Mapping[type, type] = {
    np.int_: int,
    np.int32: int,
    np.int64: int,
    np.float_: float,
    np.float32: float,
    np.float64: float,
    np.object_: object,
}


def _np_dtype_to_python_type(dtype: type) -> type:
    if dtype in NP_EQUIVALENT_TYPES_MAP:
        return NP_EQUIVALENT_TYPES_MAP[dtype]
    else:
        return dtype


def python_type_from_np_pd_dtype(dtype: type) -> type:
    return _np_dtype_to_python_type(_extract_np_dtype_from_pd(dtype))


def isinstance_compat_np_pd_dtypes(o: Any, _type: type) -> bool:
    return issubclass(python_type_from_np_pd_dtype(type(o)), python_type_from_np_pd_dtype(_type))


def isnan(value: Union[int, float]) -> bool:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Value of type {type(value)} is not supported")
    try:
        isnan_ = bool(np.isnan(value))  # numpy.bool_ --> bool
    except TypeError:  # pylint: disable=broad-except
        isnan_ = False
    return isnan_


def equal_or_nans(a: Any, b: Any) -> bool:
    a_isnan = isnan(a)
    b_isnan = isnan(b)
    if a_isnan:
        return True if b_isnan else False
    else:
        return a == b


TSequenceForRollingWindow = Union[Sequence, np.ndarray]  # np.ndarray isn't considered a Sequence but is suitable here.


def rolling_window(
    sequence: TSequenceForRollingWindow, window: int, expand: str = "neither"
) -> Tuple[TSequenceForRollingWindow, ...]:
    # TODO: Efficiency.
    if window <= 0:
        raise ValueError(f"`window` must be > 0, was {window}")
    len_ = len(sequence)
    if expand in ("left", "right", "both"):
        window = min(window, len_)
        if expand in ("left", "both"):
            slices = [(0, x) for x in range(1, window + 1)]
        else:
            slices = []
        if expand in ("right", "both"):
            slices += [(x, min(x + window, len_)) for x in range(len_)]
        else:
            slices += [(x, x + window) for x in range(len_ - window + 1)]
        slices = list(sorted(set(slices)))
    else:
        slices = [(x, x + window) for x in range(len_ - window + 1)]
    return tuple([sequence[s0:s1] for s0, s1 in slices])


def empty_df_like(like_df: pd.DataFrame) -> pd.DataFrame:
    df = like_df.iloc[:0, :].copy()  # Empty df with right dtypes etc.
    return df


def is_namedtuple(o: Any) -> bool:
    if isinstance(o, DotMap):
        return False  # Needed as the below hasattr checks will always return true for DotMap.
    # Credit for below line: https://stackoverflow.com/a/62692640
    return isinstance(o, tuple) and hasattr(o, "_asdict") and hasattr(o, "_fields")


def safe_init_dotmap(o: object) -> DotMap:
    return DotMap(o, _dynamic=False)


def split_multi_index_dataframe(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Data frame did not have a multi-index.")
    iter_index = list(df.index.levels[0])
    return (df.loc[idx, :] for idx in iter_index)


def df_eq_indicator(df: pd.DataFrame, indicator: float) -> pd.DataFrame:
    if not isnan(indicator):
        return df == indicator
    else:
        return df.isnull()
