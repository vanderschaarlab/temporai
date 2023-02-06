# pylint: disable=redefined-outer-name, unused-argument

import enum
from typing import Union

import pydantic
import pytest

import tempor.data._settings as settings
import tempor.data._types as types
from tempor.data import _check_data_container_def as check


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
def patch_check_module(patch_module):
    patch_module(
        main_module=check,
        module_vars=[
            (types, types.DataContainer, "DataContainer", Union[DummyDataContainerA, DummyDataContainerB]),
            (types, types.DataCategory, "DataCategory", DummyDataCategory),
            (types, types.ContainerFlavor, "ContainerFlavor", DummyContainerFlavor),
            (
                settings,
                settings.DATA_CATEGORY_TO_CONTAINER_FLAVORS,
                "DATA_CATEGORY_TO_CONTAINER_FLAVORS",
                DUMMY_DATA_CATEGORY_TO_CONTAINER_FLAVORS,
            ),
            (
                settings,
                settings.CONTAINER_CLASS_TO_CONTAINER_FLAVORS,
                "CONTAINER_CLASS_TO_CONTAINER_FLAVORS",
                DUMMY_CONTAINER_CLASS_TO_CONTAINER_FLAVORS,
            ),
            (settings, settings.DEFAULT_CONTAINER_FLAVOR, "DEFAULT_CONTAINER_FLAVOR", DUMMY_DEFAULT_CONTAINER_FLAVOR),
        ],
        refresh_pydantic=True,
    )


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
def test_init_success(patch_check_module, passing_combination):
    check.CheckDataContainerDefinition(
        data_category=passing_combination["data_category"],
        container_class=passing_combination["container_class"],
        container_flavor=passing_combination["container_flavor"],
    )


def test_init_fails_wrong_container_class(patch_check_module):
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
def test_init_fails_wrong_combination_data_category(patch_check_module, failing_combination):
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
def test_init_fails_wrong_combination_data_container(patch_check_module, failing_combination):
    with pytest.raises(pydantic.ValidationError, match=".*container.*"):
        check.CheckDataContainerDefinition(
            data_category=failing_combination["data_category"],
            container_class=failing_combination["container_class"],
            container_flavor=failing_combination["container_flavor"],
        )


def test_init_fails_wrong_combination_data_container_data_category(patch_check_module):
    with pytest.raises(pydantic.ValidationError, match=".*category.*container.*"):
        check.CheckDataContainerDefinition(
            data_category=DummyDataCategory.Y,  # pyright: ignore
            container_class=DummyDataContainerB,  # pyright: ignore
            container_flavor=DummyContainerFlavor.CAT_X_CONT_A_1,  # pyright: ignore
        )


def test_init_fails_no_default(patch_check_module):
    del DUMMY_DEFAULT_CONTAINER_FLAVOR[(DummyDataCategory.X, DummyDataContainerA)]
    with pytest.raises(pydantic.ValidationError, match=".*default.*"):
        check.CheckDataContainerDefinition(
            data_category=DummyDataCategory.X,  # pyright: ignore
            container_class=DummyDataContainerA,  # pyright: ignore
            container_flavor=DummyContainerFlavor.CAT_X_CONT_A_1,  # pyright: ignore
        )
