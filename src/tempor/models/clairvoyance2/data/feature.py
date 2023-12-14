# mypy: ignore-errors

import warnings
from typing import Sequence

from ..utils.common import NP_EQUIVALENT_TYPES_MAP, python_type_from_np_pd_dtype
from .constants import (
    T_CategoricalDtype,
    T_CategoricalDtype_AsTuple,
    T_ElementsObjectType_AsTuple,
    T_FeatureContainer,
    T_NumericDtype_AsTuple,
)
from .internal_utils import all_items_are_of_types

WARN_TOO_MANY_CATEGORIES_THRESHOLD = 100

PD_SERIES_OBJECT_DTYPE_ALLOWED_TYPES = T_ElementsObjectType_AsTuple  # May change.


def _infer_dtype(series: T_FeatureContainer) -> type:
    if series.dtype != object:
        return python_type_from_np_pd_dtype(series.dtype)  # type: ignore
    else:
        if all_items_are_of_types(series, str):
            return str
        else:
            return object


def _infer_categories(data: T_FeatureContainer) -> Sequence[T_CategoricalDtype]:
    unique = data.unique()
    if unique.dtype in NP_EQUIVALENT_TYPES_MAP:
        result: Sequence[T_CategoricalDtype] = tuple([NP_EQUIVALENT_TYPES_MAP[unique.dtype](x) for x in unique])
    else:
        result = tuple(unique)
    return result


class Feature:
    def __init__(self, name: str, series: T_FeatureContainer) -> None:
        self.name = name
        self.series = series

    @property
    def inferred_dtype(self) -> type:
        inferred_dtype = _infer_dtype(self.series)
        if self.series.dtype == object and inferred_dtype not in PD_SERIES_OBJECT_DTYPE_ALLOWED_TYPES:
            raise TypeError(
                f"Series of dtype object must contain homogeneous elements of one of types: "
                f"{PD_SERIES_OBJECT_DTYPE_ALLOWED_TYPES}"
            )
        return _infer_dtype(self.series)

    @property
    def numeric_compatible(self) -> bool:
        return self.inferred_dtype in T_NumericDtype_AsTuple

    @property
    def categorical_compatible(self) -> bool:
        _ = self._get_categories()  # To raise warning.
        return self.inferred_dtype in T_CategoricalDtype_AsTuple

    @property
    def binary_compatible(self) -> bool:
        cats = self._get_categories()
        return self.categorical_compatible and len(cats) <= 2

    def _get_categories(self) -> Sequence[T_CategoricalDtype]:
        categories = _infer_categories(self.series)
        if len(categories) >= WARN_TOO_MANY_CATEGORIES_THRESHOLD:
            warnings.warn(
                f"The number of categories in feature {self.name} was >={WARN_TOO_MANY_CATEGORIES_THRESHOLD}. "
                "Check this feature was intended to be used as a categorical feature",
                category=UserWarning,
            )
        return categories

    @property
    def categories(self) -> Sequence[T_CategoricalDtype]:
        if self.inferred_dtype not in T_CategoricalDtype_AsTuple:
            raise TypeError(
                f"Feature '{self.name}' is does not have the right inferred dtype ({self.inferred_dtype}) "
                "to be a categorical feature"
            )
        categories = self._get_categories()
        return categories

    def _members_repr(self) -> str:  # pragma: no cover
        return f"name={self.name}, dtype={self.inferred_dtype}"

    def _build_repr(self, members_repr: str) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({members_repr}, series={str(object.__repr__(self.series))})"

    def __repr__(self) -> str:
        return self._build_repr(members_repr=self._members_repr())

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Feature):
            return (
                self.name == __o.name
                and self.inferred_dtype == __o.inferred_dtype
                and (self.series == __o.series).all()
            )
        else:
            return False
