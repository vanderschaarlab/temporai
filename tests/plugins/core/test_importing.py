# pylint: disable=unused-argument

import os
import pathlib
from unittest.mock import Mock

import pytest

import tempor
import tempor.plugins.core._plugin as plugin_core

plugin_file_1 = """
import tempor.plugins.core._plugin as plugin_core


@plugin_core.register_plugin(name="dummy_plugin_a", category="dummy_category")
class DummyPluginA(plugin_core.Plugin):
    pass


@plugin_core.register_plugin(name="dummy_plugin_b", category="dummy_category")
class DummyPluginB(plugin_core.Plugin):
    pass

"""

plugin_file_2 = """
import tempor.plugins.core._plugin as plugin_core


@plugin_core.register_plugin(name="dummy_plugin_c", category="dummy_category")
class DummyPluginC(plugin_core.Plugin):
    pass

"""

plugin_tempor_file = """
import tempor.plugins.core._plugin as plugin_core


@plugin_core.register_plugin(name="dummy_tempor_plugin", category="dummy_category")
class DummyPluginTempor(plugin_core.Plugin):
    pass

"""


def test_import_plugins(tmp_path: pathlib.Path, patch_plugins_core_module):
    test_plugins_dir = tmp_path / tempor.import_name / "dummy_plugins"
    test_plugins_dir.mkdir(parents=True, exist_ok=True)
    with open(test_plugins_dir / "plugin_file_1.py", "w", encoding="utf8") as f:
        f.write(plugin_file_1)
    with open(test_plugins_dir / "plugin_file_2.py", "w", encoding="utf8") as f:
        f.write(plugin_file_2)
    with open(test_plugins_dir / "plugin_tempor_file.py", "w", encoding="utf8") as f:
        f.write(plugin_tempor_file)

    plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)

    init_file = os.path.join(str(test_plugins_dir), "__init__.py")
    plugin_core.importing.import_plugins(init_file)
    names = plugin_core.importing.gather_modules_names(init_file)

    assert sorted(
        [
            "dummy_category.dummy_plugin_a",
            "dummy_category.dummy_plugin_b",
            "dummy_category.dummy_plugin_c",
            "dummy_category.dummy_tempor_plugin",
        ]
    ) == sorted(plugin_core.PLUGIN_REGISTRY.keys())

    assert sorted(
        [
            "tempor.dummy_plugins.plugin_tempor_file",
            "tempor.dummy_plugins.plugin_file_2",
            "tempor.dummy_plugins.plugin_file_1",
        ]
    ) == sorted(names)


def test_import_fail(tmp_path: pathlib.Path, monkeypatch, patch_plugins_core_module):
    test_plugins_dir = tmp_path / tempor.import_name / "dummy_plugins"
    test_plugins_dir.mkdir(parents=True, exist_ok=True)
    with open(test_plugins_dir / "plugin_file_1.py", "w", encoding="utf8") as f:
        f.write(plugin_file_1)

    import importlib.util

    monkeypatch.setattr(importlib.util, "spec_from_file_location", Mock(return_value=None))

    plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)
    init_file = os.path.join(str(test_plugins_dir), "__init__.py")
    with pytest.raises(RuntimeError, match=".*[Ii]mport failed.*"):
        plugin_core.importing.import_plugins(init_file)
