import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.imputation import BaseImputer
from tempor.plugins.preprocessing.imputation.plugin_ffill import FFillImputer as plugin


def from_api() -> BaseImputer:
    return plugin_loader.get("preprocessing.imputation.ffill", random_state=123)


def from_module() -> BaseImputer:
    return plugin(random_state=123)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ffill_plugin_sanity(test_plugin: BaseImputer) -> None:
    assert test_plugin is not None
