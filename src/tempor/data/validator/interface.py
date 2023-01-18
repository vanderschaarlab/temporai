from abc import ABC, abstractmethod
from typing import List

import tempor.data as dat
import tempor.data._types as types
import tempor.data.requirements as r


class DataValidatorInterface(ABC):
    @abstractmethod
    def validate(
        self, data: types.DataContainer, requirements: List[r.DataRequirement], container_flavor: dat.ContainerFlavor
    ) -> types.DataContainer:  # pragma: no cover
        ...
