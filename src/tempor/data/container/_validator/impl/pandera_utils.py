from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd
import pandera as pa

import tempor.data._types as types
from tempor.log import logger

_DATA_FRAME_SCHEMA_INIT_ARGUMENTS = [
    "columns",
    "checks",
    "index",
    "dtype",
    "coerce",
    "strict",
    "name",
    "ordered",
    "unique",
    "report_duplicates",
    "unique_column_names",
    "title",
    "description",
]


_INDEX_INIT_ARGUMENTS = [
    "dtype",
    "checks",
    "nullable",
    "unique",
    "report_duplicates",
    "coerce",
    "name",
    "title",
    "description",
]


_MULTI_INDEX_INIT_ARGUMENTS = [
    "indexes",
    "coerce",
    "strict",
    "name",
    "ordered",
    "unique",
    "report_duplicates",
]


def _get_args(obj, arg_names: List[str]) -> dict:
    # Obtain the init arguments that correspond to go the current state of the object.
    # Try attributes with matching name:
    args = set(arg_names)
    items = {k: v for k, v in obj.__dict__.items() if k in args}
    # If any left, try attributes prepended with `_`.
    args_left = args - set(items.keys())
    _args_left = set([f"_{i}" for i in args_left])
    additional_items = {k[1:]: v for k, v in obj.__dict__.items() if k in _args_left}
    items.update(additional_items)
    return items


def update_schema(schema: pa.DataFrameSchema, /, **kwargs) -> pa.DataFrameSchema:
    items = _get_args(schema, arg_names=_DATA_FRAME_SCHEMA_INIT_ARGUMENTS)
    items.update(kwargs)
    return pa.DataFrameSchema(**items)


def update_index(index: pa.Index, **kwargs) -> pa.Index:
    items = _get_args(index, arg_names=_INDEX_INIT_ARGUMENTS)
    items.update(kwargs)
    return pa.Index(**items)


def update_multiindex(multi_index: pa.MultiIndex, **kwargs) -> pa.MultiIndex:
    items = _get_args(multi_index, arg_names=_MULTI_INDEX_INIT_ARGUMENTS)
    items.update(kwargs)
    return pa.MultiIndex(**items)


PA_DTYPE_MAP: Dict[types.Dtype, pa.DataType] = {
    bool: pa.Bool(),
    int: pa.Int(),
    float: pa.Float(),
    str: pa.String(),
    "category": pa.Category(),
    "datetime": pa.DateTime(),
}


def get_pa_dtypes(dtypes: Iterable[types.Dtype]) -> Set[pa.DataType]:
    pa_dtypes = []
    for dt in dtypes:
        if isinstance(dt, pa.DataType):
            pa_dtypes.append(dt)
        else:
            try:
                pa_dtypes.append(PA_DTYPE_MAP[dt])
            except KeyError as ex:
                raise KeyError(f"Mapping from `{dt}` to a pandera DataType not found") from ex
    return set(pa_dtypes)


def check_by_series_schema(
    series: pd.Series, series_name: str, dtypes: Set[pa.DataType], **series_schema_kwargs
) -> bool:
    # Will check that the series satisfies the SeriesSchema with at least one dtype from dtypes.
    # May pass additional SeriesSchema kwargs via series_schema_kwargs.
    logger.trace(f"Doing {series_name} dtype validation.")
    validated: List[bool] = []
    for type_ in dtypes:
        try:
            pa.SeriesSchema(type_, **series_schema_kwargs).validate(series)
            logger.trace(f"{series_name} validated? Yes: {type_}")
            validated.append(True)
            break
        except (pa.errors.SchemaError, pa.errors.SchemaErrors):
            logger.trace(f"{series_name} validated?  No: {type_}")
            validated.append(False)
    return any(validated)


def add_df_wide_checks(schema: pa.DataFrameSchema, /, *, checks_list: List[pa.Check]) -> pa.DataFrameSchema:
    schema = update_schema(schema, checks=checks_list)
    return schema


def add_all_column_checks(
    schema: pa.DataFrameSchema, /, *, dtype: Any, nullable: bool, checks_list: List[pa.Check]
) -> pa.DataFrameSchema:
    schema_out = schema.add_columns(
        {
            ".*": pa.Column(
                dtype=dtype,
                nullable=nullable,
                regex=True,
                checks=checks_list,  # type: ignore
            )
        }
    )
    assert isinstance(schema_out, pa.DataFrameSchema)
    return schema_out


def set_up_index(
    schema: pa.DataFrameSchema,
    data: pd.DataFrame,
    *,
    name: str,
    nullable: bool,
    unique: bool,
    checks_list: List[pa.Check],
) -> Tuple[pa.DataFrameSchema, pd.DataFrame]:
    if schema.index is None:
        raise ValueError("Expected DataFrameSchema Index to not be None")
    index = update_index(
        schema.index,
        nullable=nullable,
        unique=unique,
        name=name,
        checks=checks_list,
    )
    schema = update_schema(schema, index=index)
    data.index.set_names(name, inplace=True)  # Name the index.
    return schema, data


def set_up_2level_multiindex(
    schema: pa.DataFrameSchema,
    data: pd.DataFrame,
    *,
    names: Tuple[str, str],
    nullable: Tuple[bool, bool],
    unique: Tuple[str, ...],
    checks_list: Tuple[List[pa.Check], List[pa.Check]],
) -> Tuple[pa.DataFrameSchema, pd.DataFrame]:
    if schema.index is None:
        raise ValueError("Expected DataFrameSchema Index to not be None")
    if not isinstance(schema.index, pa.MultiIndex):
        raise ValueError("Expected DataFrameSchema Index to not be MultiIndex")
    if len(schema.index.indexes) != 2:
        raise ValueError("Expected DataFrameSchema Index to have 2 levels")

    index_0 = update_index(
        schema.index.indexes[0],
        name=names[0],
        nullable=nullable[0],
        checks=checks_list[0],
    )
    index_1 = update_index(
        schema.index.indexes[1],
        name=names[1],
        checks=checks_list[1],
    )

    index = update_multiindex(schema.index, indexes=[index_0, index_1], unique=unique)
    schema = update_schema(schema, index=index)
    data.index.set_names(names, inplace=True)  # Name the index.

    return schema, data


class checks:  # Functions as a "namespace" for clarity.
    forbid_multiindex_index = pa.Check(lambda df: df.index.nlevels == 1, error="MultiIndex Index not allowed")
    require_2level_multiindex_index = pa.Check(
        lambda df: df.index.nlevels == 2, error="Index must be a MultiIndex with 2 levels"
    )
    require_2level_multiindex_one_to_one = pa.Check(
        lambda df: (df.groupby(level=0).size() == 1).all(),
        error="MultiIndex Index must one-to-one correspondence for between the two levels",
    )
    forbid_multiindex_columns = pa.Check(lambda df: df.columns.nlevels == 1, error="MultiIndex Columns not allowed")

    class dynamic:
        @staticmethod
        def values_are_one_of_dtypes(dtypes: Set[types.Dtype]) -> pa.Check:
            series_name = "(Column) Values"
            error = str(f"DataFrame {series_name} dtype validation failed, must be one of: {dtypes}")
            return pa.Check(
                lambda col: check_by_series_schema(
                    pd.Series(col),
                    series_name=series_name,
                    dtypes=get_pa_dtypes(dtypes),
                    nullable=True,
                    coerce=False,  # NOTE: For simplicity "coerce-able" values are not accepted.
                ),
                error=error,
            )

        @staticmethod
        def index_is_one_of_dtypes(dtypes: Set[types.Dtype]) -> pa.Check:
            series_name = "Index"
            error = str(f"DataFrame {series_name} dtype validation failed, must be one of: {dtypes}")
            return pa.Check(
                lambda idx: check_by_series_schema(
                    pd.Series(idx),
                    series_name=series_name,
                    dtypes=get_pa_dtypes(dtypes),
                    nullable=True,
                    coerce=False,
                ),
                error=error,
            )

        @staticmethod
        def column_index_is_one_of_dtypes(dtypes: Set[types.Dtype], *, nullable: bool) -> pa.Check:
            series_name = "Column Index"
            error = str(f"DataFrame {series_name} dtype validation failed, must be one of: {dtypes}")
            return pa.Check(
                lambda df: check_by_series_schema(
                    pd.Series(df.columns),
                    series_name=series_name,
                    dtypes=get_pa_dtypes(dtypes),
                    nullable=nullable,
                    coerce=False,
                ),
                error=error,
            )
