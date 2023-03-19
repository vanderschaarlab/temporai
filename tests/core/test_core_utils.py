# pylint: disable=redefined-outer-name

import pytest

from tempor.core import utils


@pytest.fixture
def dummy_version() -> str:
    return "1.5.3"


def test_get_version(dummy_version: str):
    major, minor, patch = utils.get_version(dummy_version)
    assert (major, minor, patch) == (1, 5, 3)
