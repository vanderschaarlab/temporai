from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Type

import tempor.core
import tempor.data as dat
import tempor.data.requirements as r
import tempor.exc
from tempor.data.validator import interface
from tempor.log import log_helpers, logger

_ValidationMethod = Callable[["ValidatorImplementation", dat.DataContainer, r.DataRequirement], dat.DataContainer]


class ValidatorImplementation(interface.DataValidatorInterface, ABC):
    validation_methods: Dict[Type[r.DataRequirement], _ValidationMethod]
    _validation_records: Dict

    def __init__(self) -> None:
        super().__init__()

        # Check all data requirements are handled:
        expected_reqs = r.DATA_REQUIREMENTS[self.data_category]
        found_reqs = set(self.validation_methods.keys())
        if expected_reqs != found_reqs:
            raise TypeError(
                f"Expected {self.__class__.__name__} to have handled the following data requirements in "
                f"its validation methods {list(expected_reqs)} but found {list(found_reqs)}"
            )

    @property
    @abstractmethod
    def data_category(self) -> dat.DataCategory:  # pragma: no cover
        ...

    @abstractmethod
    def root_validate(self, data: dat.DataContainer) -> dat.DataContainer:  # pragma: no cover
        # Always do this validation.
        ...

    def validate(
        self, data: dat.DataContainer, requirements: List[r.DataRequirement], container_flavor: dat.ContainerFlavor
    ) -> dat.DataContainer:
        with log_helpers.exc_to_log():
            try:
                self._validation_records = dict()
                logger.debug("Doing root validation")
                data = self.root_validate(data)
                for req in requirements:
                    logger.debug(f"Validating requirement {req}")
                    data = self.validation_methods[type(req)](self, data, req)
                self._validation_records = dict()
            except Exception as ex:
                raise tempor.exc.DataValidationFailedException(
                    "Data validation failed, see traceback for more details"
                ) from ex
        return data


class RegisterValidation(tempor.core.RegisterMethodDecorator):
    owner_class: type = ValidatorImplementation
    registration_dict_attribute_name: str = "validation_methods"
    key_type: type = r.DataRequirement
    method_category_name: str = "validation"
