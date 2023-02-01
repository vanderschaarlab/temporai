# pylint: disable=redefined-outer-name, unused-argument

from unittest.mock import Mock

import pytest

import tempor.core.requirements as r
import tempor.data.container._requirements as dr
import tempor.data.container._validator.impl as vi
import tempor.exc

MockCategoryX = Mock()
MockCategoryY = Mock()


class DummyRequirement:
    @property
    def requirement_category(self):
        return r.RequirementCategory.DATA_CONTAINER


class DummyRequirementA(DummyRequirement):
    pass


class DummyRequirementB(DummyRequirement):
    pass


class DummyRequirementC(DummyRequirement):
    pass


@pytest.fixture
def register_dummy_requirements(monkeypatch):
    dummies = {
        MockCategoryX: {DummyRequirementA, DummyRequirementB},
        MockCategoryY: {DummyRequirementB, DummyRequirementC},
    }
    monkeypatch.setattr(dr, "DATA_CONTAINER_REQUIREMENTS_REGISTRY", dummies)


@pytest.mark.parametrize(
    "validation_methods_dict",
    [
        dict(),
        {
            DummyRequirementC: Mock(),
        },
        {
            DummyRequirementA: Mock(),
            DummyRequirementC: Mock(),
        },
    ],
)
def test_validation_implementation_wrong_methods(validation_methods_dict, register_dummy_requirements):
    with pytest.raises(TypeError, match=".*data requirements.*"):

        class TestValImpl(vi.ValidatorImplementation):
            @property
            def data_category(self):
                return MockCategoryX

            validation_methods = validation_methods_dict

            def root_validate(self, target):
                return target

        _ = TestValImpl()


def test_validation_implementation_wrong_methods_via_decorator(register_dummy_requirements):
    with pytest.raises(TypeError, match=".*data requirements.*"):

        class TestValImpl(vi.ValidatorImplementation):
            @property
            def data_category(self):
                return MockCategoryX

            @vi.RegisterValidation.register_method_for(DummyRequirementC)  # pyright: ignore
            def _(self, target, req):
                return target

            def root_validate(self, target):
                return target

        _ = TestValImpl()


def test_validate_method_success(register_dummy_requirements):
    mock_data = Mock()
    mock_req_a = DummyRequirementA()
    mock_req_b1 = DummyRequirementB()
    mock_req_b2 = DummyRequirementB()
    mock_root_validate_method = Mock()
    mock_validate_a_method = Mock()
    mock_validate_b_method = Mock()

    class TestValImpl(vi.ValidatorImplementation):
        @property
        def data_category(self):
            return MockCategoryX

        def root_validate(self, target):
            mock_root_validate_method(data=target)
            return target

        @vi.RegisterValidation.register_method_for(DummyRequirementA)  # pyright: ignore
        def _(self, target, req):
            mock_validate_a_method(data=target, req=req)
            return target

        @vi.RegisterValidation.register_method_for(DummyRequirementB)  # pyright: ignore
        def _(self, target, req):
            mock_validate_b_method(data=target, req=req)
            return target

    val = TestValImpl()
    val.validate(
        target=mock_data,
        requirements=[mock_req_a, mock_req_b1, mock_req_b2],  # pyright: ignore
        container_flavor=Mock(),
    )

    mock_root_validate_method.assert_called_once()
    mock_validate_a_method.assert_called_once()
    mock_validate_b_method.assert_called()

    mock_root_validate_method.assert_called_once_with(data=mock_data)
    mock_validate_a_method.assert_called_once_with(data=mock_data, req=mock_req_a)
    mock_validate_b_method.assert_any_call(data=mock_data, req=mock_req_b1)  # Non-last call.
    mock_validate_b_method.assert_called_with(data=mock_data, req=mock_req_b2)  # Last call.


def test_validation_exception(register_dummy_requirements):
    mock_data = Mock()
    mock_req_a = DummyRequirementA()
    mock_root_validate_method = Mock()

    class TestValImpl(vi.ValidatorImplementation):
        @property
        def data_category(self):
            return MockCategoryX

        def root_validate(self, target):
            mock_root_validate_method(data=target)
            return target

        @vi.RegisterValidation.register_method_for(DummyRequirementA)  # pyright: ignore
        def _(self, data, req):
            raise ValueError("My inner error")

        @vi.RegisterValidation.register_method_for(DummyRequirementB)  # pyright: ignore
        def _(self, data, req):
            # Do nothing.
            return data

    val = TestValImpl()

    with pytest.raises(tempor.exc.DataValidationFailedException) as excinfo:
        val.validate(target=mock_data, requirements=[mock_req_a], container_flavor=Mock())  # pyright: ignore
    assert "My inner error" in str(excinfo.getrepr())
