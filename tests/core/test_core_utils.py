# pylint: disable=redefined-outer-name

import pytest
from typing_extensions import Literal

from tempor.core import utils


@pytest.fixture
def dummy_version() -> str:
    return "1.5.3"


def test_get_version(dummy_version: str):
    major, minor, patch = utils.get_version(dummy_version)
    assert (major, minor, patch) == (1, 5, 3)


@pytest.mark.parametrize(
    "version, compare, expected",
    [
        ((1, 6), (1, 5), True),
        ((1, 5), (1, 5), True),
        ((1, 4), (1, 5), False),
        ((2, 4), (1, 5), True),
        ((2, 5), (1, 5), True),
        ((2, 6), (1, 5), True),
        ((0, 6), (1, 5), False),
        ((0, 5), (1, 5), False),
        ((0, 4), (1, 5), False),
    ],
)
def test_version_above_incl(version, compare, expected):
    assert utils.version_above_incl(version=version, above_incl=compare) is expected


@pytest.mark.parametrize(
    "version, compare, expected",
    [
        ((1, 6), (1, 5), False),
        ((1, 5), (1, 5), False),
        ((1, 4), (1, 5), True),
        ((2, 4), (1, 5), False),
        ((2, 5), (1, 5), False),
        ((2, 6), (1, 5), False),
        ((0, 6), (1, 5), True),
        ((0, 5), (1, 5), True),
        ((0, 4), (1, 5), True),
    ],
)
def test_version_below_excl(version, compare, expected):
    assert utils.version_below_excl(version=version, below_excl=compare) is expected


class TestEnsureLiteralMatchesDictKeys:
    @pytest.mark.parametrize(
        "lit, d",
        [
            (Literal["A", "B", "C"], {"A": None, "B": None, "C": None}),
            (Literal["A", "B", "C"], {"A": None, "B": None, "C": None}),
            (Literal["A"], {"A": None}),
        ],
    )
    def test_passes(self, lit, d):
        utils.ensure_literal_matches_dict_keys(lit, d)

    @pytest.mark.parametrize(
        "lit, d",
        [
            (Literal["A", "B", "C", "D"], {"A": None, "B": None, "C": None}),
            (Literal["A", "B", "X"], {"A": None, "B": None, "C": None}),
            (Literal["A", "B"], {"A": None, "B": None, "C": None}),
            (Literal["A"], {"B": None}),
            (Literal["A"], dict()),
        ],
    )
    def test_fails(self, lit, d):
        with pytest.raises(TypeError, match=".*MyLit.*MyDict.*"):
            utils.ensure_literal_matches_dict_keys(lit, d, literal_name="MyLit", dict_name="MyDict")

    @pytest.mark.parametrize(
        "lit, d, expect",
        [
            (Literal["A", "B", "C", "D"], {"A": None, "B": None, "C": None}, set(["D"])),
            (Literal["A", "C", "D"], {"A": None, "B": None, "C": None, "D": None}, set(["D"])),
            (Literal["A", "C", "D", "X"], {"A": None, "B": None, "C": None, "D": None}, set(["D", "X"])),
        ],
    )
    def test_fails_difference(self, lit, d, expect):
        with pytest.raises(TypeError, match=f".*{list(str(expect))}.*"):
            utils.ensure_literal_matches_dict_keys(lit, d)
