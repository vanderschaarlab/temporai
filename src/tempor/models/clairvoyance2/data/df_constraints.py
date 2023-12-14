# mypy: ignore-errors

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.common import python_type_from_np_pd_dtype
from .internal_utils import all_items_are_of_types

# NOTE: Obtained from https://pbpython.com/pandas_dtypes.html. May not be fully accurate.
PD_ACCEPTABLE_EQUIVALENT_DTYPES = (
    object,
    int,
    float,
    bool,
    np.datetime64,
    pd.Timedelta,
    pd.Categorical,
    np.int32,
    np.int64,
    np.float64,
    np.double,
)


# TODO: Unit test.
def cast_to_index_constraints_dtypes(dtypes_tuple: Tuple[type, ...]) -> Tuple[type, ...]:
    python_equivalents = [python_type_from_np_pd_dtype(t) for t in dtypes_tuple]
    casts = [t if t in PD_ACCEPTABLE_EQUIVALENT_DTYPES else object for t in python_equivalents]
    return tuple(set(casts))


@dataclass
class IndexConstraints:
    types: Optional[Sequence[pd.Index]] = None
    dtypes: Optional[Sequence[type]] = None
    dtype_object_constrain_types: Optional[Sequence[type]] = None
    enforce_monotonic_increasing: bool = False
    enforce_unique: bool = False
    enforce_not_multi_index: bool = False


@dataclass
class ElementConstraints:
    dtypes: Optional[Sequence[type]] = None
    dtype_object_constrain_types: Optional[Sequence[type]] = None
    enforce_homogenous_type_per_column: Optional[bool] = None


@dataclass
class Constraints:
    on_index: Optional[IndexConstraints] = None
    on_columns: Optional[IndexConstraints] = None
    on_elements: Optional[ElementConstraints] = None


class ConstraintsChecker:
    def __init__(self, constraints: Constraints) -> None:
        self.constraints = constraints

    def check(self, df: pd.DataFrame) -> bool:
        if self.constraints.on_index is not None:
            self._check_index_or_columns(df.index, self.constraints.on_index, "index")
        if self.constraints.on_columns is not None:
            self._check_index_or_columns(df.columns, self.constraints.on_columns, "columns")
        if self.constraints.on_elements is not None:
            self._check_elements(df, self.constraints.on_elements)
        return True

    @staticmethod
    def _get_all_object_columns(df: pd.DataFrame) -> Iterable:
        return (col for col, dtype in df.dtypes.items() if dtype == object)

    @staticmethod
    def _check_index_or_columns(
        index_or_columns: pd.Index, constraints: IndexConstraints, index_or_columns_str: str
    ) -> None:
        if constraints.types is not None and len(constraints.types) > 0:
            if not isinstance(index_or_columns, tuple(constraints.types)):  # type: ignore
                raise TypeError(
                    f"DataFrame {index_or_columns_str} must be one of types: {constraints.types}. "
                    f"Was found to be of type: {type(index_or_columns)}."
                )
        if constraints.dtypes is not None and len(constraints.dtypes) > 0:
            if python_type_from_np_pd_dtype(index_or_columns.dtype) not in constraints.dtypes:  # type: ignore
                raise TypeError(
                    f"DataFrame {index_or_columns_str} dtype must be one of: {constraints.dtypes}. "
                    f"Was found to be of dtype: {index_or_columns.dtype}"
                )
        if constraints.dtype_object_constrain_types is not None and len(constraints.dtype_object_constrain_types) > 0:
            if index_or_columns.dtype == object:
                if any(not isinstance(r, tuple(constraints.dtype_object_constrain_types)) for r in index_or_columns):
                    raise TypeError(
                        f"DataFrame {index_or_columns_str} of dtype object must be constrained "
                        f"to the following types: {constraints.dtype_object_constrain_types}. "
                        f"Check dtype of each element of DataFrame {index_or_columns_str}"
                    )
        if constraints.enforce_monotonic_increasing:
            if not index_or_columns.is_monotonic_increasing:
                raise TypeError(f"DataFrame {index_or_columns_str} must be monotonic increasing")
        if constraints.enforce_not_multi_index:
            if not isinstance(index_or_columns, pd.MultiIndex) is False:
                raise TypeError(f"DataFrame {index_or_columns_str} must not be multi-index")
        if constraints.enforce_unique:
            if not index_or_columns.is_unique:
                raise TypeError(f"DataFrame {index_or_columns_str} must be unique")

    @staticmethod
    def _check_elements(df: pd.DataFrame, constraints: ElementConstraints) -> None:
        if constraints.dtypes is not None and len(constraints.dtypes) > 0:
            if not all(python_type_from_np_pd_dtype(dtype) in constraints.dtypes for dtype in df.dtypes.values):
                raise TypeError(
                    f"DataFrame elements must be limited to dtypes: {constraints.dtypes}. "
                    "Check by calling `.dtype()` on your DataFrame."
                )
        if constraints.dtype_object_constrain_types and len(constraints.dtype_object_constrain_types) > 0:
            if any(
                not all_items_are_of_types(df[col], tuple(constraints.dtype_object_constrain_types))
                for col in ConstraintsChecker._get_all_object_columns(df)
            ):
                raise TypeError(
                    f"DataFrame elements of dtype object must be constrained "
                    f"to the following types: {constraints.dtype_object_constrain_types}. "
                    "Check elements of columns of dtype object."
                )
        if constraints.enforce_homogenous_type_per_column:
            if len(df) > 0 and any(
                not all_items_are_of_types(df[col], type(df[col].iat[0]))
                for col in ConstraintsChecker._get_all_object_columns(df)
            ):
                raise TypeError(
                    "DataFrame elements must be of homogenous type in every column, "
                    "including the type of elements in columns of dtype object."
                )
