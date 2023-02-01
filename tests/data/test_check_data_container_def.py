# pylint: disable=redefined-outer-name, unused-argument

import copy
import enum
from typing import Union

import pydantic
import pytest

import tempor.data._settings as settings
import tempor.data._types as types
from tempor.data.container import _check_data_container_def as check


class DummyDataCategory(enum.Enum):
    X = enum.auto()
    Y = enum.auto()


class DummyDataContainerA:
    pass


class DummyDataContainerB:
    pass


class DummyDataContainerC:
    pass


class DummyContainerFlavor(enum.Enum):
    CAT_X_CONT_A_1 = enum.auto()
    CAT_X_CONT_A_2 = enum.auto()
    CAT_X_CONT_B = enum.auto()
    CAT_Y_CONT_B_1 = enum.auto()
    CAT_Y_CONT_B_2 = enum.auto()


DUMMY_DATA_CATEGORY_TO_CONTAINER_FLAVORS = {
    DummyDataCategory.X: {
        DummyContainerFlavor.CAT_X_CONT_A_1,
        DummyContainerFlavor.CAT_X_CONT_A_2,
        DummyContainerFlavor.CAT_X_CONT_B,
    },
    DummyDataCategory.Y: {
        DummyContainerFlavor.CAT_Y_CONT_B_1,
        DummyContainerFlavor.CAT_Y_CONT_B_2,
    },
}


DUMMY_CONTAINER_CLASS_TO_CONTAINER_FLAVORS = {
    DummyDataContainerA: {
        DummyContainerFlavor.CAT_X_CONT_A_1,
        DummyContainerFlavor.CAT_X_CONT_A_2,
    },
    DummyDataContainerB: {
        DummyContainerFlavor.CAT_X_CONT_B,
        DummyContainerFlavor.CAT_Y_CONT_B_1,
        DummyContainerFlavor.CAT_Y_CONT_B_2,
    },
}

DUMMY_DEFAULT_CONTAINER_FLAVOR = {
    (DummyDataCategory.X, DummyDataContainerA): DummyContainerFlavor.CAT_X_CONT_A_1,
    (DummyDataCategory.X, DummyDataContainerB): DummyContainerFlavor.CAT_X_CONT_B,
    (DummyDataCategory.Y, DummyDataContainerB): DummyContainerFlavor.CAT_Y_CONT_B_1,
}


@pytest.fixture
def patch_module(monkeypatch):
    # Reload the module under test while also refreshing aspects of pydantic as needed.
    # See: https://github.com/streamlit/streamlit/issues/3218#issuecomment-1050647471

    undo = dict()
    undo["DataContainer"] = copy.deepcopy(types.DataContainer)
    undo["DataCategory"] = copy.deepcopy(types.DataCategory)
    undo["ContainerFlavor"] = copy.deepcopy(types.ContainerFlavor)
    undo["DATA_CATEGORY_TO_CONTAINER_FLAVORS"] = copy.deepcopy(settings.DATA_CATEGORY_TO_CONTAINER_FLAVORS)
    undo["CONTAINER_CLASS_TO_CONTAINER_FLAVORS"] = copy.deepcopy(settings.CONTAINER_CLASS_TO_CONTAINER_FLAVORS)
    undo["DEFAULT_CONTAINER_FLAVOR"] = copy.deepcopy(settings.DEFAULT_CONTAINER_FLAVOR)
    monkeypatch.setattr(types, "DataContainer", Union[DummyDataContainerA, DummyDataContainerB])
    monkeypatch.setattr(types, "DataCategory", DummyDataCategory)
    monkeypatch.setattr(types, "ContainerFlavor", DummyContainerFlavor)
    monkeypatch.setattr(settings, "DATA_CATEGORY_TO_CONTAINER_FLAVORS", DUMMY_DATA_CATEGORY_TO_CONTAINER_FLAVORS)
    monkeypatch.setattr(settings, "CONTAINER_CLASS_TO_CONTAINER_FLAVORS", DUMMY_CONTAINER_CLASS_TO_CONTAINER_FLAVORS)
    monkeypatch.setattr(settings, "DEFAULT_CONTAINER_FLAVOR", DUMMY_DEFAULT_CONTAINER_FLAVOR)

    pydantic.class_validators._FUNCS.clear()  # pylint: disable=protected-access
    import importlib

    importlib.reload(check)

    yield

    # Teardown - must manually undo the monkeypatching to what it should be, for other tests to run correctly:

    types.DataContainer = undo["DataContainer"]
    types.DataCategory = undo["DataCategory"]
    types.ContainerFlavor = undo["ContainerFlavor"]
    settings.DATA_CATEGORY_TO_CONTAINER_FLAVORS = undo["DATA_CATEGORY_TO_CONTAINER_FLAVORS"]
    settings.CONTAINER_CLASS_TO_CONTAINER_FLAVORS = undo["CONTAINER_CLASS_TO_CONTAINER_FLAVORS"]
    settings.DEFAULT_CONTAINER_FLAVOR = undo["DEFAULT_CONTAINER_FLAVOR"]

    pydantic.class_validators._FUNCS.clear()  # pylint: disable=protected-access
    importlib.reload(check)


@pytest.mark.parametrize(
    "passing_combination",
    [
        {
            "data_category": DummyDataCategory.X,
            "container_class": DummyDataContainerA,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_A_1,
        },
        {
            "data_category": DummyDataCategory.X,
            "container_class": DummyDataContainerA,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_A_2,
        },
        {
            "data_category": DummyDataCategory.X,
            "container_class": DummyDataContainerB,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_B,
        },
        {
            "data_category": DummyDataCategory.Y,
            "container_class": DummyDataContainerB,
            "container_flavor": DummyContainerFlavor.CAT_Y_CONT_B_1,
        },
        {
            "data_category": DummyDataCategory.Y,
            "container_class": DummyDataContainerB,
            "container_flavor": DummyContainerFlavor.CAT_Y_CONT_B_2,
        },
    ],
)
def test_init_success(patch_module, passing_combination):
    check.CheckDataContainerDefinition(
        data_category=passing_combination["data_category"],
        container_class=passing_combination["container_class"],
        container_flavor=passing_combination["container_flavor"],
    )


def test_init_fails_wrong_container_class(patch_module):
    with pytest.raises(pydantic.ValidationError, match=".*container_class.*"):
        check.CheckDataContainerDefinition(
            data_category=DummyDataCategory.X,  # pyright: ignore
            container_class=DummyDataContainerC,  # pyright: ignore
            container_flavor=DummyContainerFlavor.CAT_X_CONT_A_1,  # pyright: ignore
        )


@pytest.mark.parametrize(
    "failing_combination",
    [
        {
            "data_category": DummyDataCategory.X,
            "container_class": DummyDataContainerB,
            "container_flavor": DummyContainerFlavor.CAT_Y_CONT_B_1,
        },
        {
            "data_category": DummyDataCategory.X,
            "container_class": DummyDataContainerB,
            "container_flavor": DummyContainerFlavor.CAT_Y_CONT_B_2,
        },
        {
            "data_category": DummyDataCategory.Y,
            "container_class": DummyDataContainerA,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_A_1,
        },
        {
            "data_category": DummyDataCategory.Y,
            "container_class": DummyDataContainerA,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_A_2,
        },
        {
            "data_category": DummyDataCategory.Y,
            "container_class": DummyDataContainerB,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_B,
        },
    ],
)
def test_init_fails_wrong_combination_data_category(patch_module, failing_combination):
    with pytest.raises(pydantic.ValidationError, match=".*category.*"):
        check.CheckDataContainerDefinition(
            data_category=failing_combination["data_category"],
            container_class=failing_combination["container_class"],
            container_flavor=failing_combination["container_flavor"],
        )


@pytest.mark.parametrize(
    "failing_combination",
    [
        {
            "data_category": DummyDataCategory.X,
            "container_class": DummyDataContainerB,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_A_1,
        },
        {
            "data_category": DummyDataCategory.X,
            "container_class": DummyDataContainerB,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_A_2,
        },
        {
            "data_category": DummyDataCategory.X,
            "container_class": DummyDataContainerA,
            "container_flavor": DummyContainerFlavor.CAT_X_CONT_B,
        },
        {
            "data_category": DummyDataCategory.Y,
            "container_class": DummyDataContainerA,
            "container_flavor": DummyContainerFlavor.CAT_Y_CONT_B_1,
        },
        {
            "data_category": DummyDataCategory.Y,
            "container_class": DummyDataContainerA,
            "container_flavor": DummyContainerFlavor.CAT_Y_CONT_B_2,
        },
    ],
)
def test_init_fails_wrong_combination_data_container(patch_module, failing_combination):
    with pytest.raises(pydantic.ValidationError, match=".*container.*"):
        check.CheckDataContainerDefinition(
            data_category=failing_combination["data_category"],
            container_class=failing_combination["container_class"],
            container_flavor=failing_combination["container_flavor"],
        )


def test_init_fails_wrong_combination_data_container_data_category(patch_module):
    with pytest.raises(pydantic.ValidationError, match=".*category.*container.*"):
        check.CheckDataContainerDefinition(
            data_category=DummyDataCategory.Y,  # pyright: ignore
            container_class=DummyDataContainerB,  # pyright: ignore
            container_flavor=DummyContainerFlavor.CAT_X_CONT_A_1,  # pyright: ignore
        )


def test_init_fails_no_default(patch_module):
    del DUMMY_DEFAULT_CONTAINER_FLAVOR[(DummyDataCategory.X, DummyDataContainerA)]
    with pytest.raises(pydantic.ValidationError, match=".*default.*"):
        check.CheckDataContainerDefinition(
            data_category=DummyDataCategory.X,  # pyright: ignore
            container_class=DummyDataContainerA,  # pyright: ignore
            container_flavor=DummyContainerFlavor.CAT_X_CONT_A_1,  # pyright: ignore
        )
