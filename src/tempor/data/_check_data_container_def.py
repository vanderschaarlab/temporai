import typing
from typing import Type

import pydantic
import pydantic.dataclasses

from . import _settings as settings
from . import _types as types


@pydantic.dataclasses.dataclass(frozen=True)
class CheckDataContainerDefinition:
    data_category: types.DataCategory
    container_class: Type[types.DataContainer]
    container_flavor: types.ContainerFlavor

    @pydantic.validator("container_class")
    def check_data_container_type_in_possible_types(cls, v):  # pylint: disable=no-self-argument
        possible_types = typing.get_args(types.DataContainer)
        if not issubclass(v, possible_types):
            raise TypeError(f"`container_class` must be one of the data container types {possible_types} but was {v}")
        else:
            return v

    @pydantic.root_validator(skip_on_failure=True)
    def container_class_matches_data_category(cls, values):  # pylint: disable=no-self-argument
        data_category = values.get("data_category")
        container_class = values.get("container_class")
        container_flavor = values.get("container_flavor")
        if container_flavor not in settings.DATA_CATEGORY_TO_CONTAINER_FLAVORS[data_category]:
            raise ValueError(
                f"Data category {data_category} supports container flavors "
                f"{list(settings.DATA_CATEGORY_TO_CONTAINER_FLAVORS[data_category])} but container flavor "
                f"{container_flavor} found"
            )
        if container_flavor not in settings.CONTAINER_CLASS_TO_CONTAINER_FLAVORS[container_class]:
            raise ValueError(
                f"Data container class {container_class} supports container flavors "
                f"{list(settings.CONTAINER_CLASS_TO_CONTAINER_FLAVORS[container_class])} but container flavor "
                f"{container_flavor} found"
            )
        if (data_category, container_class) not in settings.DEFAULT_CONTAINER_FLAVOR.keys():
            raise ValueError(
                "Default container flavor for the (data category, container class) combination "
                f"{(data_category, container_class)} has not been set in settings, defaults are as follows:\n"
                f"{settings.DEFAULT_CONTAINER_FLAVOR}"
            )
        return values
