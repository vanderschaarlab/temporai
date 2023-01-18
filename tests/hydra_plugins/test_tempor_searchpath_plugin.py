import hydra
from hydra.core import global_hydra, plugins
from hydra.plugins import search_path_plugin

from hydra_plugins.tempor_searchpath_plugin import TemporSearchPathPlugin


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    assert TemporSearchPathPlugin.__name__ in [
        x.__name__ for x in plugins.Plugins.instance().discover(search_path_plugin.SearchPathPlugin)
    ]


def test_config_installed() -> None:
    with hydra.initialize(version_base=None):
        config_loader = global_hydra.GlobalHydra.instance().config_loader()
    assert "config" in config_loader.get_group_options("tempor")
