# pylint: disable=redefined-outer-name, unused-argument, protected-access

from typing import Dict
from unittest.mock import Mock

import pytest

import tempor.data.container._requirements as dr

MockCategoryA = Mock()
MockCategoryB = Mock()
MockCategoryC = Mock()


@pytest.fixture
def empty_out_data_requirements(monkeypatch):
    empty: Dict = {
        MockCategoryA: set(),
        MockCategoryB: set(),
        MockCategoryC: set(),
    }
    monkeypatch.setattr(dr, "DATA_CONTAINER_REQUIREMENTS_REGISTRY", empty)


def test_create_data_requirements_success(empty_out_data_requirements):
    class SomeRequirement(dr.DataContainerRequirement):
        data_categories = {MockCategoryA}

    class AnotherRequirement(dr.DataContainerRequirement):
        data_categories = {MockCategoryA, MockCategoryB}

    class YetAnotherRequirement(dr.DataContainerRequirement):
        data_categories = {MockCategoryA, MockCategoryB, MockCategoryC}

    reqs_registry = dr.DATA_CONTAINER_REQUIREMENTS_REGISTRY

    assert type(SomeRequirement) == dr._DataContainerRequirementMeta
    assert reqs_registry[MockCategoryA] == {SomeRequirement, AnotherRequirement, YetAnotherRequirement}
    assert reqs_registry[MockCategoryB] == {AnotherRequirement, YetAnotherRequirement}
    assert reqs_registry[MockCategoryC] == {YetAnotherRequirement}


def test_disallow_empty_data_categories(empty_out_data_requirements):
    with pytest.raises(TypeError, match=".*at least one.*"):

        class EmptyCategories(dr.DataContainerRequirement):  # pylint: disable=unused-variable
            data_categories = {}  # pyright: ignore


def test_wrong_names(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(dr._DataContainerRequirementMeta, "_base_class_name", "WrongNameBaseClass")

        with pytest.raises(TypeError, match=".*WrongNameBaseClass.*"):

            class BaseClassWasWrongNameFailure(dr.DataContainerRequirement):  # pylint: disable=unused-variable
                pass

    with monkeypatch.context() as m:
        m.setattr(dr._DataContainerRequirementMeta, "_data_categories_attr_name", "wrong_attr")

        with pytest.raises(TypeError, match=".*wrong_att*"):

            class CategoriesAttrWrongNameFailure(dr.DataContainerRequirement):  # pylint: disable=unused-variable
                pass
