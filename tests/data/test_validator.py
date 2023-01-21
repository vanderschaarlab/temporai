# pylint: disable=redefined-outer-name, unused-argument

import enum
from typing import Dict, Tuple, Union
from unittest.mock import Mock

import pytest

import tempor.data._check_data_container_def as check
import tempor.data._types as types
import tempor.data.validator as v

MockCategoryA = Mock()


@pytest.fixture
def dummy_implementation():
    return Mock()


class DummyValImpl:
    # Dummy ValidationImplementation class.
    pass


class DummyContainerA:
    pass


class DummyContainerB:
    pass


DummyContainerType = Union[DummyContainerA, DummyContainerB]


class DummyContainerFlavor(enum.Enum):
    CONT_A_1 = enum.auto()
    CONT_A_2 = enum.auto()
    CONT_B_1 = enum.auto()


mock_check = Mock()

DummyDef = Tuple[type, DummyContainerFlavor]


@pytest.fixture
def set_dummy_container_types(monkeypatch):
    monkeypatch.setattr(types, "DataContainer", DummyContainerType)


@pytest.fixture
def set_dummy_check(monkeypatch):
    monkeypatch.setattr(check, "CheckDataContainerDefinition", mock_check)


def test_validator_init_success(dummy_implementation, set_dummy_container_types, set_dummy_check):
    class TestDataValidator(v.DataValidator):
        @property
        def data_category(self):
            return MockCategoryA

        @property
        def supports_implementations_for(self) -> Tuple[DummyDef, ...]:
            return ((DummyContainerA, DummyContainerFlavor.CONT_A_1), (DummyContainerB, DummyContainerFlavor.CONT_B_1))

        def _register_implementations(self) -> Dict[DummyDef, DummyValImpl]:
            return {
                (DummyContainerA, DummyContainerFlavor.CONT_A_1): dummy_implementation,
                (DummyContainerB, DummyContainerFlavor.CONT_B_1): dummy_implementation,
            }

    _ = TestDataValidator()

    mock_check.assert_any_call(  # Non-last call.
        data_category=MockCategoryA, container_class=DummyContainerA, container_flavor=DummyContainerFlavor.CONT_A_1
    )
    mock_check.assert_called_with(  # Last call.
        data_category=MockCategoryA, container_class=DummyContainerB, container_flavor=DummyContainerFlavor.CONT_B_1
    )


def test_validator_validate_success(dummy_implementation, set_dummy_container_types, set_dummy_check):
    class TestDataValidator(v.DataValidator):
        @property
        def data_category(self):
            return MockCategoryA

        @property
        def supports_implementations_for(self) -> Tuple[DummyDef, ...]:
            return ((DummyContainerA, DummyContainerFlavor.CONT_A_1), (DummyContainerB, DummyContainerFlavor.CONT_B_1))

        def _register_implementations(self) -> Dict[DummyDef, DummyValImpl]:
            return {
                (DummyContainerA, DummyContainerFlavor.CONT_A_1): dummy_implementation,
                (DummyContainerB, DummyContainerFlavor.CONT_B_1): dummy_implementation,
            }

    validator = TestDataValidator()
    validator.validate(
        data=DummyContainerA(),  # pyright: ignore
        requirements=[],
        container_flavor=DummyContainerFlavor.CONT_A_1,  # pyright: ignore
    )
    validator.validate(
        data=DummyContainerB(),  # pyright: ignore
        requirements=[],
        container_flavor=DummyContainerFlavor.CONT_B_1,  # pyright: ignore
    )


def test_validator_validate_fails_not_data_container_type(
    dummy_implementation, set_dummy_container_types, set_dummy_check
):
    class TestDataValidator(v.DataValidator):
        @property
        def data_category(self):
            return MockCategoryA

        @property
        def supports_implementations_for(self) -> Tuple[DummyDef, ...]:
            return ((DummyContainerA, DummyContainerFlavor.CONT_A_1),)

        def _register_implementations(self) -> Dict[DummyDef, DummyValImpl]:
            return {
                (DummyContainerA, DummyContainerFlavor.CONT_A_1): dummy_implementation,
            }

    validator = TestDataValidator()

    with pytest.raises(TypeError, match=".supported types.*"):
        validator.validate(
            data="string_is_wrong_type",  # pyright: ignore
            requirements=[],
            container_flavor=DummyContainerFlavor.CONT_A_1,  # pyright: ignore
        )


def test_instantiation_success():
    _ = v.StaticDataValidator()
    _ = v.TimeSeriesDataValidator()
    _ = v.EventDataValidator()
