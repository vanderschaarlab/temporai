# pylint: disable=redefined-outer-name, unused-argument

from unittest.mock import Mock

import pytest

import tempor.data.requirements as r

MockCategoryA = Mock()
MockCategoryB = Mock()
MockCategoryC = Mock()


@pytest.fixture
def empty_out_data_requirements(monkeypatch):
    empty = {
        MockCategoryA: set(),
        MockCategoryB: set(),
        MockCategoryC: set(),
    }
    monkeypatch.setattr(r, "DATA_REQUIREMENTS", empty)


def test_create_data_requirements_success(empty_out_data_requirements):
    class SomeRequirement(r.DataRequirement):
        categories = {MockCategoryA}

    class AnotherRequirement(r.DataRequirement):
        categories = {MockCategoryA, MockCategoryB}

    class YetAnotherRequirement(r.DataRequirement):
        categories = {MockCategoryA, MockCategoryB, MockCategoryC}

    reqs_registry = r.DATA_REQUIREMENTS

    assert type(SomeRequirement) == r.DataRequirementMeta
    assert reqs_registry[MockCategoryA] == {SomeRequirement, AnotherRequirement, YetAnotherRequirement}
    assert reqs_registry[MockCategoryB] == {AnotherRequirement, YetAnotherRequirement}
    assert reqs_registry[MockCategoryC] == {YetAnotherRequirement}


def test_disallow_empty_categories(empty_out_data_requirements):
    with pytest.raises(TypeError, match=".*at least one.*"):

        class EmptyCategories(r.DataRequirement):  # pylint: disable=unused-variable
            categories = {}  # pyright: ignore


def test_wrong_names(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(r.DataRequirementMeta, "_base_class_name", "WrongNameBaseClass")

        with pytest.raises(TypeError, match=".*WrongNameBaseClass.*"):

            class BaseClassWasWrongNameFailure(r.DataRequirement):  # pylint: disable=unused-variable
                pass

    with monkeypatch.context() as m:
        m.setattr(r.DataRequirementMeta, "_categories_attr_name", "wrong_attr")

        with pytest.raises(TypeError, match=".*wrong_att*"):

            class CategoriesAttrWrongNameFailure(r.DataRequirement):  # pylint: disable=unused-variable
                pass
