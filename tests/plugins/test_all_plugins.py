"""Tests that will be automatically run for all plugins."""

import pytest

from tempor.plugins import plugin_loader

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
