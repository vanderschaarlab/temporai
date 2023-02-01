from unittest.mock import Mock

import pydantic
import pytest

from tempor.core import requirements as r

MyReqCategory = Mock()


def test_init_success():
    class MyRequirement(r.Requirement):
        name = "my_requirement"

        @property
        def requirement_category(self):
            return MyReqCategory

    _ = MyRequirement(definition="some value")


def test_init_fails_no_name():
    class MyRequirement(r.Requirement):
        @property
        def requirement_category(self):
            return MyReqCategory

    with pytest.raises(ValueError, match=".*name.*"):
        _ = MyRequirement(definition="some value")


def test_init_definition_type_success():
    class MyRequirement(r.Requirement):
        name = "my_requirement"
        definition: float

        @property
        def requirement_category(self):
            return MyReqCategory

    _ = MyRequirement(definition=1.5)


def test_init_definition_type_fails():
    class MyRequirement(r.Requirement):
        name = "my_requirement"
        definition: float

        @property
        def requirement_category(self):
            return MyReqCategory

    with pytest.raises(ValueError, match=".*definition.*"):
        _ = MyRequirement(definition="incompatible")


def test_init_extra_validator_success():
    mock_call = Mock()

    @pydantic.validator("definition")
    def extra_validator(cls, v):  # pylint: disable=unused-argument
        mock_call(value=v)
        return v

    class MyRequirement(r.Requirement):
        name = "my_requirement"
        definition: float
        validator_methods = [extra_validator]

        @property
        def requirement_category(self):
            return MyReqCategory

    _ = MyRequirement(definition=1.5)

    mock_call.assert_called_once_with(value=1.5)


def test_init_extra_validator_fail():
    @pydantic.validator("definition")
    def extra_validator(cls, v):  # pylint: disable=unused-argument
        if v != "foo":
            raise ValueError("Not foo")
        return v

    class MyRequirement(r.Requirement):
        name = "my_requirement"
        definition: str
        validator_methods = [extra_validator]

        @property
        def requirement_category(self):
            return MyReqCategory

    with pytest.raises(ValueError, match=".*Not foo.*"):
        _ = MyRequirement(definition="bar")


def test_validator_validate_success():
    mock_validate = Mock()
    mock_target = Mock()

    class MyValidator(r.RequirementValidator):
        @property
        def supported_requirement_category(self):
            return MyReqCategory

        def _validate(self, *args, **kwargs):
            mock_validate(*args, **kwargs)

    class MyRequirement(r.Requirement):
        name = "my_requirement"

        @property
        def requirement_category(self):
            return MyReqCategory

    requirements = (MyRequirement("A"), MyRequirement("B"))
    validator = MyValidator()

    validator.validate(target=mock_target, requirements=requirements)

    mock_validate.assert_called_once_with(target=mock_target, requirements=requirements)


def test_validator_validate_fails_not_supported_category():
    mock_validate = Mock()
    mock_target = Mock()

    MyOtherCategory = Mock()

    class MyValidator(r.RequirementValidator):
        @property
        def supported_requirement_category(self):
            return MyReqCategory

        def _validate(self, *args, **kwargs):
            mock_validate(*args, **kwargs)

    class MyOtherRequirement(r.Requirement):
        name = "my_requirement"

        @property
        def requirement_category(self):
            return MyOtherCategory

    requirements = (MyOtherRequirement("A"), MyOtherRequirement("B"))
    validator = MyValidator()

    with pytest.raises(ValueError):
        validator.validate(target=mock_target, requirements=requirements)

    mock_validate.assert_not_called()
