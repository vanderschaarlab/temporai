import abc
from typing import ClassVar, Dict, List, Set, Type

import pydantic

import tempor.core.requirements as r
import tempor.data._settings as settings
import tempor.data._types as types

DATA_CONTAINER_REQUIREMENTS_REGISTRY: Dict[types.DataCategory, Set[Type["DataContainerRequirement"]]] = {
    types.DataCategory.STATIC: set(),
    types.DataCategory.TIME_SERIES: set(),
    types.DataCategory.EVENT: set(),
}


class _DataContainerRequirementMeta(abc.ABCMeta):
    _base_class_name: str = "DataContainerRequirement"
    _data_categories_attr_name: str = "data_categories"

    def __new__(mcls, name, bases, namespace, /, *args, **kwargs):  # pyright: ignore
        data_req_cls: Type["DataContainerRequirement"] = super().__new__(
            mcls, name, bases, namespace, *args, **kwargs  # type: ignore
        )

        name = data_req_cls.__name__  # type: ignore
        mro_names = [cls.__name__ for cls in data_req_cls.mro()]  # type: ignore
        if name != mcls._base_class_name and mcls._base_class_name not in mro_names:
            raise TypeError(
                f"A class with metaclass {mcls.__name__} should either be the base class named {mcls._base_class_name} "
                f"or inherit from the base class named {mcls._base_class_name}"
            )
        if not hasattr(data_req_cls, mcls._data_categories_attr_name):
            raise TypeError(
                f"A class with metaclass {mcls.__name__} should have an attribute "
                f"named {mcls._data_categories_attr_name}"
            )

        if not data_req_cls.__name__ == mcls._base_class_name and not data_req_cls.data_categories:  # type: ignore
            raise TypeError(
                f"{data_req_cls.__name__} must define at least one {mcls._data_categories_attr_name}"  # pyright: ignore
            )

        # Register each data requirement:
        for cat in data_req_cls.data_categories:
            DATA_CONTAINER_REQUIREMENTS_REGISTRY[cat].add(data_req_cls)

        return data_req_cls


class DataContainerRequirement(
    r.Requirement,
    metaclass=_DataContainerRequirementMeta,
):
    data_categories: ClassVar[Set[types.DataCategory]] = set()

    @property
    def requirement_category(self) -> r.RequirementCategory:
        return r.RequirementCategory.DATA_CONTAINER


@pydantic.validator("definition")
def _value_dtypes_check_supported(cls, v):  # pylint: disable=unused-argument
    unsupported = set(v) - set(settings.DATA_SETTINGS.value_dtypes)
    if unsupported:
        raise TypeError(
            f"Value dtypes must only include supported dtypes {list(settings.DATA_SETTINGS.value_dtypes)} "
            f"but {list(unsupported)} found"
        )
    return v


class ValueDTypes(DataContainerRequirement):
    name = "value_dtypes"
    definition: List[types.Dtype]
    validator_methods = [_value_dtypes_check_supported]

    data_categories: ClassVar[Set[types.DataCategory]] = {
        types.DataCategory.STATIC,
        types.DataCategory.TIME_SERIES,
        types.DataCategory.EVENT,
    }


class AllowMissing(DataContainerRequirement):
    name = "allow_missing"
    definition: bool

    data_categories: ClassVar[Set[types.DataCategory]] = {
        types.DataCategory.STATIC,
        types.DataCategory.TIME_SERIES,
        types.DataCategory.EVENT,
    }
