"""Tests that will be automatically run for all plugins."""
import re

import pytest

from tempor import plugin_loader
from tempor.methods.core._base_estimator import EmptyParamsDefinition

PLUGIN_FQNS = plugin_loader.list_full_names(plugin_type="method")


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.parametrize("plugin_fqn", PLUGIN_FQNS)
def test_sample_hyperparameters(plugin_fqn):
    PluginCls = plugin_loader.get_class(plugin_fqn)
    for repeat in range(10):  # pylint: disable=unused-variable
        args = PluginCls.sample_hyperparameters()
        PluginCls(**args)


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
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
        r"'.*plugin_type='"
        f"{plugin.plugin_type}"
        r"'.*params=.?\{.*\}.*\)",
        repr_,
        re.S | re.DOTALL,
    )
    if PluginCls.ParamsDefinition is not None and not isinstance(PluginCls.ParamsDefinition, EmptyParamsDefinition):
        params_dict = dict(plugin.params)
        for key in params_dict.keys():
            assert f"'{key}':" in repr_
