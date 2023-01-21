from typing import ClassVar, Dict, List, Set, Type

import pydantic

import tempor.data as dat

DATA_REQUIREMENTS: Dict[dat.DataCategory, Set[Type["DataRequirement"]]] = {
    dat.DataCategory.STATIC: set(),
    dat.DataCategory.TIME_SERIES: set(),
    dat.DataCategory.EVENT: set(),
}


class DataRequirementMeta(pydantic.main.ModelMetaclass, type):
    _base_class_name: str = "DataRequirement"
    _categories_attr_name: str = "categories"

    def __new__(mcls, name, bases, namespace, /, **kwargs):  # pyright: ignore
        data_req_cls: Type["DataRequirement"] = super().__new__(mcls, name, bases, namespace, **kwargs)  # type: ignore

        name = data_req_cls.__name__  # type: ignore
        mro_names = [cls.__name__ for cls in data_req_cls.mro()]  # type: ignore
        if name != mcls._base_class_name and mcls._base_class_name not in mro_names:
            raise TypeError(
                f"A class with metaclass {mcls.__name__} should either be the base class named {mcls._base_class_name} "
                f"or inherit from the base class named {mcls._base_class_name}"
            )
        if not hasattr(data_req_cls, mcls._categories_attr_name):
            raise TypeError(
                f"A class with metaclass {mcls.__name__} should have an attribute "
                f"named {mcls._categories_attr_name}"
            )

        if not data_req_cls.__name__ == mcls._base_class_name and not data_req_cls.categories:  # type: ignore
            raise TypeError(
                f"{data_req_cls.__name__} must define at least one {mcls._categories_attr_name}"  # pyright: ignore
            )

        # Register each data requirement:
        for cat in data_req_cls.categories:
            DATA_REQUIREMENTS[cat].add(data_req_cls)

        return data_req_cls


class DataRequirement(pydantic.BaseModel, metaclass=DataRequirementMeta):  # pylint: disable=no-member
    categories: ClassVar[Set[dat.DataCategory]] = set()

    class Config:
        arbitrary_types_allowed = True


class ValueDTypes(DataRequirement):
    value_dtypes: List[dat.Dtype]

    @pydantic.validator("value_dtypes")
    def check_supported_value_dtypes(cls, v):  # pylint: disable=no-self-argument
        unsupported = set(v) - set(dat.DATA_SETTINGS.value_dtypes)
        if unsupported:
            raise TypeError(
                f"Value dtypes must only include supported dtypes {list(dat.DATA_SETTINGS.value_dtypes)} "
                f"but {list(unsupported)} found"
            )
        return v

    categories: ClassVar[Set[dat.DataCategory]] = {
        dat.DataCategory.STATIC,
        dat.DataCategory.TIME_SERIES,
        dat.DataCategory.EVENT,
    }


class AllowMissing(DataRequirement):
    allow_missing: bool

    categories: ClassVar[Set[dat.DataCategory]] = {
        dat.DataCategory.STATIC,
        dat.DataCategory.TIME_SERIES,
        dat.DataCategory.EVENT,
    }
