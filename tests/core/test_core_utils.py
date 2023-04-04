import pytest
from typing_extensions import Literal

from tempor.core import utils


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
