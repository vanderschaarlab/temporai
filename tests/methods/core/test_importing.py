# pylint: disable=unused-argument

import os
import pathlib
from unittest.mock import Mock

import pytest

import tempor
import tempor.methods.core._plugin as plugin_core

plugin_file_1 = """
import tempor.methods.core._plugin as plugin_core


@plugin_core.register_plugin(name="dummy_plugin_a", category="dummy_category")
class DummyPluginA(plugin_core.Plugin):
    pass


@plugin_core.register_plugin(name="dummy_plugin_b", category="dummy_category")
class DummyPluginB(plugin_core.Plugin):
    pass

"""

plugin_file_2 = """
import tempor.methods.core._plugin as plugin_core


@plugin_core.register_plugin(name="dummy_plugin_c", category="dummy_category")
class DummyPluginC(plugin_core.Plugin):
    pass

"""

plugin_tempor_file = """
import tempor.methods.core._plugin as plugin_core


@plugin_core.register_plugin(name="dummy_tempor_plugin", category="dummy_category")
class DummyPluginTempor(plugin_core.Plugin):
    pass

"""

plugin_different_type_file = """
import tempor.methods.core._plugin as plugin_core


@plugin_core.register_plugin(
    name="dummy_different_type_plugin",
    category="dummy_different_type_category",
    plugin_type="my_plugin_type",
)
class DummyPluginDifferentCategory(plugin_core.Plugin):
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
    with open(test_plugins_dir / "plugin_different_type_file.py", "w", encoding="utf8") as f:
        f.write(plugin_different_type_file)

    plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)
    plugin_core.register_plugin_category(
        "dummy_different_type_category", expected_class=plugin_core.Plugin, plugin_type="my_plugin_type"
    )

    init_file = os.path.join(str(test_plugins_dir), "__init__.py")
    plugin_core.importing.import_plugins(init_file)
    names = plugin_core.importing.gather_modules_names(init_file)

    assert sorted(
        [
            f"[{plugin_core.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_plugin_a",
            f"[{plugin_core.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_plugin_b",
            f"[{plugin_core.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_plugin_c",
            f"[{plugin_core.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_tempor_plugin",
            "[my_plugin_type].dummy_different_type_category.dummy_different_type_plugin",
        ]
    ) == sorted(plugin_core.PLUGIN_REGISTRY.keys())

    assert sorted(
        [
            "tempor.dummy_plugins.plugin_different_type_file",
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
