from typing import Callable, Dict, Sequence, Type

import pydantic

import tempor.core
import tempor.core.requirements as r
import tempor.data._types as types
import tempor.exc
from tempor.log import log_helpers, logger

from . import _bundle


class DataBundleRequirement(r.Requirement):
    @property
    def requirement_category(self) -> r.RequirementCategory:
        return r.RequirementCategory.DATA_BUNDLE


@pydantic.validator("definition")
def _parse_to_set(cls, v):  # pylint: disable=unused-argument
    return set(v)


class DataPresent(DataBundleRequirement):
    name = "data_present"
    definition: Sequence[types.SamplesAttributes]
    validator_methods = [_parse_to_set]


_ValidationMethod = Callable[
    ["_DataBundleValidatorBase", _bundle.DataBundle, DataBundleRequirement], _bundle.DataBundle
]


class _DataBundleValidatorBase(r.RequirementValidator):
    validation_methods: Dict[Type[DataBundleRequirement], _ValidationMethod]

    @property
    def supported_requirement_category(self) -> r.RequirementCategory:
        return r.RequirementCategory.DATA_BUNDLE

    def _validate(  # pylint: disable=arguments-differ
        self,
        target: _bundle.DataBundle,
        *,
        requirements: Sequence[DataBundleRequirement],
        **kwargs,
    ):
        logger.debug(f"Running {_bundle.DataBundle.__name__} validation")
        with log_helpers.exc_to_log():
            try:
                for req in requirements:
                    logger.debug(f"Running {_bundle.DataBundle.__name__} validation for requirement: {req}")
                    self.validation_methods[type(req)](self, target, req)
            except Exception as ex:
                raise tempor.exc.DataValidationFailedException(
                    f"{_bundle.DataBundle.__name__} validation failed, see traceback for more details"
                ) from ex
        return target


class _RegisterValidation(tempor.core.RegisterMethodDecorator):
    owner_class: type = _DataBundleValidatorBase
    registration_dict_attribute_name: str = "validation_methods"
    key_type: type = DataBundleRequirement
    method_category_name: str = "validation"


class DataBundleValidator(_DataBundleValidatorBase):
    @_RegisterValidation.register_method_for(DataPresent)
    def _(self, target: _bundle.DataBundle, req: DataBundleRequirement) -> _bundle.DataBundle:
        assert isinstance(req, DataPresent)
        for samples_attribute in req.definition:
            samples = getattr(target, samples_attribute)
            if samples is None:
                raise ValueError(f"Expected the following data Samples to be defined: {samples_attribute}")
        return target
