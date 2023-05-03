"""Tests that will be automatically run for all plugins."""
import re

import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.core._base_estimator import EmptyParamsDefinition

PLUGIN_FQNS = plugin_loader.list_fqns()


@pytest.mark.parametrize("plugin_fqn", PLUGIN_FQNS)
def test_init_success(plugin_fqn):
    PluginCls = plugin_loader.get_class(plugin_fqn)
    PluginCls()  # Should successfully initialize with all default params.


@pytest.mark.parametrize("plugin_fqn", PLUGIN_FQNS)
def test_sample_hyperparameters(plugin_fqn):
    PluginCls = plugin_loader.get_class(plugin_fqn)
    for repeat in range(10):  # pylint: disable=unused-variable
        args = PluginCls.sample_hyperparameters()
        PluginCls(**args)


@pytest.mark.parametrize("plugin_fqn", PLUGIN_FQNS)
def test_repr(plugin_fqn):
    PluginCls = plugin_loader.get_class(plugin_fqn)
    plugin = PluginCls()
    repr_ = str(plugin)

    assert re.search(
        f"^{PluginCls.__name__}"
        r"\(.*name='"
        f"{plugin.name}"
        r"'.*category='"
        f"{plugin.category}"
        r"'.*params=.?\{.*\}.*\)",
        repr_,
        re.S | re.DOTALL,
    )
    if PluginCls.ParamsDefinition is not None and not isinstance(PluginCls.ParamsDefinition, EmptyParamsDefinition):
        params_dict = dict(plugin.params)
        for key in params_dict.keys():
            assert f"'{key}':" in repr_
