from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

import pandas as pd
import pandera as pa

from tempor.log import logger

from . import data_typing

pa_major, pa_minor, *_ = [int(v) for v in pa.__version__.split(".")]


_PA_DF_SCHEMA_INIT_PARAMETERS = [
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


_PA_INDEX_INIT_PARAMETERS = [
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


_PA_MULTI_INDEX_INIT_PARAMETERS = [
    "indexes",
    "coerce",
    "strict",
    "name",
    "ordered",
    "unique",
]

if pa_major == 0 and pa_minor < 14:
    # Before v0.14, pandera API had an extra parameter `report_duplicates`.
    _PA_MULTI_INDEX_INIT_PARAMETERS.append("report_duplicates")


def _get_pa_init_args(pa_object: Any, param_names: List[str]) -> Dict[str, Any]:
    """A helper method for updating `pandera` objects dynamically.

    Get values of items in `pa_object`'s `__dict__`, specified by `param_names`.
    `param_names` should contain names (`str`) of the `__init__` parameters of the `pa_object`.

    Algorithm:
    - Will attempt to get by `param_names` item.
    - If not found, will attempt to get by `param_names` item prepended with `_`.
    - If an `arg_name` item isn't found, it is ignored.

    Args:
        pa_object (Any): `pandera` object.
        param_names (List[str]): list of `pandera` object's `__init__` parameters.

    Returns:
        Dict[str, Any]: dictionary mapping `pa_object`'s `__init__` parameter names to their current values.
    """
    # Try attributes with matching name:
    args = set(param_names)
    items = {k: v for k, v in pa_object.__dict__.items() if k in args}
    # If any left, try attributes prepended with `_`.
    args_left = args - set(items.keys())
    _args_left = set([f"_{i}" for i in args_left])
    additional_items = {k[1:]: v for k, v in pa_object.__dict__.items() if k in _args_left}
    items.update(additional_items)
    return items


def update_schema(schema: pa.DataFrameSchema, **kwargs) -> pa.DataFrameSchema:
    items = _get_pa_init_args(schema, param_names=_PA_DF_SCHEMA_INIT_PARAMETERS)
    items.update(kwargs)
    return pa.DataFrameSchema(**items)


def update_index(index: pa.Index, **kwargs) -> pa.Index:
    items = _get_pa_init_args(index, param_names=_PA_INDEX_INIT_PARAMETERS)
    items.update(kwargs)
    return pa.Index(**items)


def update_multiindex(multi_index: pa.MultiIndex, **kwargs) -> pa.MultiIndex:
    items = _get_pa_init_args(multi_index, param_names=_PA_MULTI_INDEX_INIT_PARAMETERS)
    items.update(kwargs)
    return pa.MultiIndex(**items)


PA_DTYPE_MAP: Dict[data_typing.Dtype, pa.DataType] = {
    bool: pa.Bool(),
    int: pa.Int(),
    float: pa.Float(),
    str: pa.String(),
    "category": pa.Category(),
    "datetime": pa.DateTime(),
}
"""A mapping from dtype specified as `Dtype` to a `pandera.DataType`.
"""


def get_pa_dtypes(dtypes: Iterable[data_typing.Dtype]) -> List[pa.DataType]:
    """Return a `set` of `pandera.DataType` corresponding to `dtypes`. Raises `KeyError` If not found."""
    pa_dtypes = []
    for dt in dtypes:
        if isinstance(dt, pa.DataType):
            # If item in `dtypes` already a `pandera.Dtype`, pass it through.
            pa_dtypes.append(dt)
        else:
            try:
                pa_dtypes.append(PA_DTYPE_MAP[dt])
            except KeyError as ex:
                raise KeyError(f"Mapping from `{dt}` to a pandera DataType not found") from ex
    return list(set(pa_dtypes))


def check_by_series_schema(series: pd.Series, series_name: str, dtypes: List[pa.DataType], **kwargs) -> bool:
    """Will check that `series` satisfies a `SeriesSchema` with at least one dtype from `dtypes`.
    May pass additional `SeriesSchema` kwargs via `kwargs`.
    """
    logger.trace(f"Doing {series_name} dtype validation.")
    validated: List[bool] = []
    for type_ in set(dtypes):
        try:
            pa.SeriesSchema(type_, **kwargs).validate(series)
            logger.trace(f"{series_name} validated? Yes: {type_}")
            validated.append(True)
            break
        except (pa.errors.SchemaError, pa.errors.SchemaErrors):  # pyright: ignore
            logger.trace(f"{series_name} validated?  No: {type_}")
            validated.append(False)
    return any(validated)


def add_df_checks(schema: pa.DataFrameSchema, *, checks_list: List[pa.Check]) -> pa.DataFrameSchema:
    schema = update_schema(schema, checks=checks_list)
    return schema


def add_regex_column_checks(
    schema: pa.DataFrameSchema, *, regex: str = ".*", dtype: Any, nullable: bool, checks_list: List[pa.Check]
) -> pa.DataFrameSchema:
    """Update `schema` with checks specified in `checks_list`, applied to all columns specified by `regex`.
    `dtype` and `nullable` can also be specified and will apply to all columns.
    """
    schema_out = schema.add_columns(
        {
            regex: pa.Column(
                dtype=dtype,
                nullable=nullable,
                regex=True,
                checks=checks_list,  # type: ignore
            )
        }
    )
    if TYPE_CHECKING:  # pragma: no cover
        assert isinstance(schema_out, pa.DataFrameSchema)  # nosec B101
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
    """Update `schema.index` (`pandera.Index`) with `name`, `nullable`, ... schema settings.

    In addition, set the index name of `data` (`pandas.DataFrame`) to `name`.

    Returns the schema and the dataframe.
    """
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
    """Update `schema.index` (`pandera.MultiIndex`), which is expected to have 2 levels, with `name`, `nullable`, ...
    schema settings.

    In addition, set the index name of `data` (`pandas.DataFrame`) to `name`.

    Returns the schema and the dataframe.
    """
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


class checks:
    """Namespace containing reusable `pandera.Check`s."""

    forbid_multiindex_index = pa.Check(
        lambda df: df.index.nlevels == 1,
        error="MultiIndex Index not allowed",
    )
    forbid_multiindex_columns = pa.Check(
        lambda df: df.columns.nlevels == 1,
        error="MultiIndex Columns not allowed",
    )
    require_2level_multiindex_index = pa.Check(
        lambda df: df.index.nlevels == 2,
        error="Index must be a MultiIndex with 2 levels",
    )
    require_2level_multiindex_one_to_one = pa.Check(
        lambda df: (df.groupby(level=0).size() == 1).all(),
        error="MultiIndex Index must one-to-one correspondence for between the two levels",
    )
    require_element_len_2 = pa.Check(
        lambda x: len(x) == 2,
        element_wise=True,
        error="Each item must contain a sequence of length 2",
    )

    class configurable:
        """Namespace containing functions to get configurable `pandera.Check`s."""

        @staticmethod
        def values_satisfy_dtypes(dtypes: List[data_typing.Dtype]) -> pa.Check:
            series_name = "Values"
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
        def index_satisfies_dtypes(dtypes: List[data_typing.Dtype]) -> pa.Check:
            series_name = "Index"
            error = str(f"DataFrame {series_name} dtype validation failed, must be one of: {dtypes}")
            return pa.Check(
                lambda idx: check_by_series_schema(
                    pd.Series(idx),
                    series_name=series_name,
                    dtypes=get_pa_dtypes(dtypes),
                    nullable=False,
                    coerce=False,
                ),
                error=error,
            )

        @staticmethod
        def column_index_satisfies_dtypes(dtypes: List[data_typing.Dtype], *, nullable: bool) -> pa.Check:
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
