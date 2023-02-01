import abc
from typing import Callable, Dict, Sequence, Type

import tempor.core
import tempor.data._types as types
import tempor.data.container._requirements as dr
import tempor.exc
from tempor.data.container._validator import interface
from tempor.log import log_helpers, logger

_ValidationMethod = Callable[
    ["ValidatorImplementation", types.DataContainer, dr.DataContainerRequirement], types.DataContainer
]


class ValidatorImplementation(interface.DataValidatorInterface, abc.ABC):
    validation_methods: Dict[Type[dr.DataContainerRequirement], _ValidationMethod]
    _validation_records: Dict

    def __init__(self) -> None:
        super().__init__()

        # Check all data requirements are handled:
        expected_reqs = dr.DATA_CONTAINER_REQUIREMENTS_REGISTRY[self.data_category]
        found_reqs = set(self.validation_methods.keys())
        if expected_reqs != found_reqs:
            raise TypeError(
                f"Expected {self.__class__.__name__} to have handled the following data requirements in "
                f"its validation methods {list(expected_reqs)} but found {list(found_reqs)}"
            )

    @property
    @abc.abstractmethod
    def data_category(self) -> types.DataCategory:  # pragma: no cover
        ...

    @abc.abstractmethod
    def root_validate(self, target: types.DataContainer) -> types.DataContainer:  # pragma: no cover
        # Always do this validation.
        ...

    def _validate(
        self,
        target: types.DataContainer,
        *,
        requirements: Sequence[dr.DataContainerRequirement],
        container_flavor: types.ContainerFlavor,
        **kwargs,
    ) -> types.DataContainer:
        with log_helpers.exc_to_log():
            try:
                self._validation_records = dict()
                logger.debug("Doing root validation")
                target = self.root_validate(target)
                for req in requirements:
                    logger.debug(f"Validating requirement {req}")
                    target = self.validation_methods[type(req)](self, target, req)
                self._validation_records = dict()
            except Exception as ex:
                raise tempor.exc.DataValidationFailedException(
                    "Data validation failed, see traceback for more details"
                ) from ex
        return target


class RegisterValidation(tempor.core.RegisterMethodDecorator):
    owner_class: type = ValidatorImplementation
    registration_dict_attribute_name: str = "validation_methods"
    key_type: type = dr.DataContainerRequirement
    method_category_name: str = "validation"
