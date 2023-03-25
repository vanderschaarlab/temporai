import dataclasses
from typing import List

import pydantic
import pytest

from tempor.core import pydantic_utils


def test_is_pydantic_dataclass():
    @dataclasses.dataclass
    class MyDataclass:
        a: str = "string"
        b: int = 2

    MyPydanticDataclass = pydantic.dataclasses.dataclass(MyDataclass)

    assert pydantic_utils.is_pydantic_dataclass(MyDataclass) is False
    assert pydantic_utils.is_pydantic_dataclass(MyPydanticDataclass) is True


def test_pydantic_dataclass_edge_case_fail():
    @dataclasses.dataclass
    class MyDataclass:
        a: str = "string"
        b: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])

    pydantic.dataclasses.dataclass(MyDataclass)

    try:
        # At least as of 1.10.7 this will cause TypeError.
        pydantic.dataclasses.dataclass(MyDataclass)  # TypeError.
    except TypeError:
        # Will just pass rather than do pytest.raises(TypeError), in case this is fixed in a future version.
        pass


def test_make_pydantic_dataclass_workaround():
    @dataclasses.dataclass
    class MyDataclass:
        a: str = "string"
        b: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])

    MyPydanticDataclass = pydantic_utils.make_pydantic_dataclass(MyDataclass)

    # Check behaves as expected:
    MyPydanticDataclass(a="abc", b=[10, 11, 12])
    with pytest.raises(ValueError):
        MyPydanticDataclass(a="abc", b="should_be_list")

    MyPydanticDataclass = pydantic_utils.make_pydantic_dataclass(MyDataclass)  # No TypeError.

    # Check behaves as expected:
    MyPydanticDataclass(a="abc", b=[10, 11, 12])
    with pytest.raises(ValueError):
        MyPydanticDataclass(a="abc", b="should_be_list")
