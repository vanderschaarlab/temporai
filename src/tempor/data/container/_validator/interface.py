import abc
from typing import Sequence

import tempor.core.requirements as r
import tempor.data._types as types
import tempor.data.container._requirements as dr


class DataValidatorInterface(r.RequirementValidator, abc.ABC):
    @abc.abstractmethod
    def _validate(  # type: ignore  # pylint: disable=arguments-differ
        self,
        target: types.DataContainer,
        *,
        requirements: Sequence[dr.DataContainerRequirement],
        container_flavor: types.ContainerFlavor,
        **kwargs,
    ) -> types.DataContainer:  # pragma: no cover
        ...

    @property
    def supported_requirement_category(self) -> r.RequirementCategory:
        return r.RequirementCategory.DATA_CONTAINER
