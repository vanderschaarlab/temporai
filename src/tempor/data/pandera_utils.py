import sys
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandera as pa
import pandera.dtypes as pa_dtypes
import pandera.engines.pandas_engine as pd_engine
from packaging.version import Version

import tempor.core.utils

from . import data_typing

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

if Version(pa.__version__) < Version("0.14"):
    # Before v0.14, pandera API had an extra parameter `report_duplicates`.
    _PA_MULTI_INDEX_INIT_PARAMETERS.append("report_duplicates")


def _get_pa_init_args(pa_object: Any, param_names: List[str]) -> Dict[str, Any]:
    """A helper method for updating `pandera` objects dynamically.

    Get values of items in ``pa_object``'s ``__dict__``, specified by ``param_names`` .
    ``param_names`` should contain names (`str`) of the ``__init__`` parameters of the ``pa_object``.

    Algorithm:
    - Will attempt to get by ``param_names`` item.
    - If not found, will attempt to get by ``param_names`` item prepended with ``_``.
    - If an ``arg_name`` item isn't found, it is ignored.

    Args:
        pa_object (Any): `pandera` object.
        param_names (List[str]): list of `pandera` object's ``__init__`` parameters.

    Returns:
        Dict[str, Any]: dictionary mapping ``pa_object`` 's ``__init__`` parameter names to their current values.
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
"""A mapping from dtype specified as :obj:`~tempor.data.data_typing.Dtype` to a `pandera.DataType`.
"""


def get_pa_dtypes(dtypes: Iterable[data_typing.Dtype]) -> List[pa.DataType]:
    """Return a list of `pandera.DataType` corresponding to ``dtypes``. Raises `KeyError` If not found."""
    pa_dtypes_ = []
    for dt in dtypes:
        if isinstance(dt, pa.DataType):
            # If item in `dtypes` already an instance of `pandera.DataType`, pass it through.
            dt_add = dt
        elif hasattr(dt, "__mro__") and issubclass(dt, pa.DataType):  # type: ignore
            # If item in `dtypes` a `pandera.DataType` class, pass it through as an instance.
            dt_add = dt()  # type: ignore
        else:
            try:
                dt_add = PA_DTYPE_MAP[dt]
            except KeyError as ex:
                raise KeyError(f"Mapping from `{dt}` to a pandera DataType not found") from ex
        pa_dtypes_.append(dt_add)
        if sys.platform == "win32":
            # Pandera .Int()/.Float() do not appear to correctly validate on Windows, unless one specifically
            # provides the bytes.
            if dt_add == pa.Int():
                pa_dtypes_.extend([pa.Int8(), pa.Int16(), pa.Int32(), pa.Int64()])
            if dt_add == pa.Float():
                pa_dtypes_.extend([pa.Float16(), pa.Float32(), pa.Float64()])
    return list(set(pa_dtypes_))


class UnionDtype(pd_engine.DataType):
    """Extend `pandera` ``DataType`` s with a custom ``UnionDtype``, which will function similarly to ``Union``.

    See `pandera` ``DataType`` [guide](https://pandera.readthedocs.io/en/stable/dtypes.html) for details.

    In this case, rather than wrapping the extension ``DataType`` with ``register_dtype`` and ``immutable`` decorators,
    we apply these directly to the class returned by ``__class_getitem__``, which dynamically creates the union
    specified with its dtypes. In this way, `pandera`'s ``pandas`` engine correctly registers each new kind of union
    as a different dtype.
    """

    union_dtypes: List
    """The list of types in the union."""
    type: Any
    """The string representation of the data type, which will be, e.g., shown in exceptions."""

    name: str
    """The string representation of the data type used for `repr`."""

    @classmethod
    def __class_getitem__(cls, item):
        """Allows for setting union types like ``UnionDtype[dtype, ...]``.

        Acceptable ``dtype`` s are: `pandera.DataType` (as a class or instance) or the keys of
        `~tempor.data.pandera_utils.PA_DTYPE_MAP`.
        """
        if not tempor.core.utils.is_iterable(item):
            item = [item]
        union_dtypes = get_pa_dtypes(item)
        union_dtypes = sorted(union_dtypes, key=str)  # For consistency: `item` can get captured in random order.
        repr_union_dtypes = str([str(t) for t in union_dtypes]).replace("'", "")
        name = f"{cls.__name__}{repr_union_dtypes}"

        cls_ = type(name, (UnionDtype,), dict())
        cls_.union_dtypes = union_dtypes  # type: ignore
        cls_.type = name  # type: ignore
        cls_.name = name  # type: ignore

        return pd_engine.Engine.register_dtype(pa_dtypes.immutable(cls_))  # type: ignore

    def __repr__(self) -> str:
        return self.name

    def check(
        self,
        pandera_dtype: pa_dtypes.DataType,
        data_container=None,
    ) -> Union[bool, Iterable[bool]]:
        """Checks whether the ``pandera_dtype`` and optionally ``data_container`` satisfy at least one the union's
        ``union_dtypes``.

        Args:
            pandera_dtype (pa_dtypes.DataType):
                The data type received as part of the check/validation.
            data_container (PandasObject, optional):
                The data container received as part of the check/validation. Defaults to `None`.

        Returns:
            Union[bool, Iterable[bool]]:
                A `bool` stating whether the data type is satisfied, or an iterable thereof\
                (for each item in the ``data_container``).
        """
        for union_dtype in self.union_dtypes:
            validated = pd_engine.Engine.dtype(union_dtype).check(pandera_dtype, data_container)
            if data_container is None:
                # Only in case of direct type comparison, we also need to check via the union_dtype.check(pandera_dtype)
                # method, making sure that pandera_dtype is an DataType instance not class.
                if hasattr(pandera_dtype, "__mro__"):
                    pandera_dtype = pandera_dtype()  # type: ignore
                validated = validated or union_dtype.check(pandera_dtype)
            if tempor.core.utils.is_iterable(validated):
                validated = all(validated)  # type: ignore
            if validated:
                if data_container is None:
                    return True
                else:
                    return np.full_like(data_container, True, dtype=bool)

        if data_container is None:
            return False
        else:
            return np.full_like(data_container, False, dtype=bool)

    def coerce(self, data_container):
        """The ``coerce`` method is not supported and will throw a `NotImplementedError`."""
        raise NotImplementedError(f"`coerce` not supported by {self.__class__.__name__}")


def init_schema(data: pd.DataFrame, **kwargs) -> pa.DataFrameSchema:
    schema = pa.infer_schema(data)
    if TYPE_CHECKING:  # pragma: no cover
        assert isinstance(schema, pa.DataFrameSchema)  # nosec B101
    schema = update_schema(schema, **kwargs)
    return schema


def add_df_checks(schema: pa.DataFrameSchema, *, checks_list: List[pa.Check]) -> pa.DataFrameSchema:
    schema = update_schema(schema, checks=checks_list)
    return schema


def add_regex_column_checks(
    schema: pa.DataFrameSchema,
    *,
    regex: str = ".*",
    dtype: Any,
    nullable: bool,
    checks_list: Optional[List[pa.Check]] = None,
) -> pa.DataFrameSchema:
    """Update ``schema`` with checks specified in ``checks_list``, applied to all columns specified by ``regex``.
    ``dtype`` and ``nullable`` can also be specified and will apply to all columns.
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
    dtype: Any,
    name: str,
    nullable: bool,
    unique: bool,
    coerce: bool,
    checks_list: Optional[List[pa.Check]] = None,
) -> Tuple[pa.DataFrameSchema, pd.DataFrame]:
    """Update ``schema.index`` (`pandera.Index`) with ``dtype``, ``name``, ``nullable``, ... schema settings.

    In addition, set the index name of ``data`` (`pandas.DataFrame`) to ``name``.

    Returns the schema and the dataframe.
    """
    if schema.index is None:
        raise ValueError("Expected DataFrameSchema Index to not be None")
    index = update_index(
        schema.index,
        dtype=dtype,
        nullable=nullable,
        unique=unique,
        name=name,
        checks=checks_list,
        coerce=coerce,
    )
    schema = update_schema(schema, index=index)
    data.index.set_names(name, inplace=True)  # Name the index.
    return schema, data


def set_up_2level_multiindex(
    schema: pa.DataFrameSchema,
    data: pd.DataFrame,
    *,
    dtypes: Tuple[Any, Any],
    names: Tuple[str, str],
    nullable: Tuple[bool, bool],
    coerce: bool,
    unique: Tuple[str, ...],
    checks_list: Optional[Tuple[List[pa.Check], List[pa.Check]]] = None,
) -> Tuple[pa.DataFrameSchema, pd.DataFrame]:
    """Update ``schema.index`` (`pandera.MultiIndex`), which is expected to have 2 levels, with `dtypes```, ``names``,
    ``nullable``, ... schema settings.

    In addition, set the index name of ``data`` (`pandas.DataFrame`) to ``name``.

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
        dtype=dtypes[0],
        name=names[0],
        nullable=nullable[0],
        coerce=coerce,
        checks=checks_list[0] if checks_list is not None else None,
    )
    index_1 = update_index(
        schema.index.indexes[1],
        dtype=dtypes[1],
        name=names[1],
        coerce=coerce,
        checks=checks_list[1] if checks_list is not None else None,
    )

    index = update_multiindex(schema.index, indexes=[index_0, index_1], unique=unique)
    schema = update_schema(schema, index=index)
    data.index.set_names(names, inplace=True)  # Name the index.

    return schema, data


class checks:
    """Namespace containing reusable `pandera.Check` s."""

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
        """Namespace containing functions to get configurable `pandera.Check` s."""

        @staticmethod
        def column_index_satisfies_dtype(dtype: Any, *, nullable: bool) -> pa.Check:
            series_name = "Column Index"
            error = str(f"DataFrame {series_name} dtype validation failed, must be of type: {dtype}")

            def _check(df: pd.DataFrame) -> bool:
                pa.SeriesSchema(
                    dtype,
                    name=series_name,
                    nullable=nullable,
                    coerce=False,
                ).validate(pd.Series(df.columns, name=series_name))
                return True

            return pa.Check(_check, error=error)
