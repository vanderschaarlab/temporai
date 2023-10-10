# pylint: disable=unused-argument

import os
import pathlib
from unittest.mock import Mock

import pytest

import tempor
from tempor.core import plugins
from tempor.core.plugins import plugin_typing

plugin_file_1 = """
from tempor.core import plugins


@plugins.register_plugin(name="dummy_plugin_a", category="dummy_category")
class DummyPluginA(plugins.Plugin):
    pass


@plugins.register_plugin(name="dummy_plugin_b", category="dummy_category")
class DummyPluginB(plugins.Plugin):
    pass

"""

plugin_file_2 = """
from tempor.core import plugins


@plugins.register_plugin(name="dummy_plugin_c", category="dummy_category")
class DummyPluginC(plugins.Plugin):
    pass

"""

plugin_tempor_file = """
from tempor.core import plugins


@plugins.register_plugin(name="dummy_tempor_plugin", category="dummy_category")
class DummyPluginTempor(plugins.Plugin):
    pass

"""

plugin_different_type_file = """
from tempor.core import plugins


@plugins.register_plugin(
    name="dummy_different_type_plugin",
    category="dummy_different_type_category",
    plugin_type="my_plugin_type",
)
class DummyPluginDifferentCategory(plugins.Plugin):
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

    plugins.register_plugin_category("dummy_category", expected_class=plugins.Plugin)
    plugins.register_plugin_category(
        "dummy_different_type_category", expected_class=plugins.Plugin, plugin_type="my_plugin_type"
    )

    init_file = os.path.join(str(test_plugins_dir), "__init__.py")
    plugins.importing.import_plugins(init_file)
    names = plugins.importing.gather_modules_names(init_file)

    assert sorted(
        [
            f"[{plugin_typing.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_plugin_a",
            f"[{plugin_typing.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_plugin_b",
            f"[{plugin_typing.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_plugin_c",
            f"[{plugin_typing.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_tempor_plugin",
            "[my_plugin_type].dummy_different_type_category.dummy_different_type_plugin",
        ]
    ) == sorted(plugins.PLUGIN_REGISTRY.keys())

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

    plugins.register_plugin_category("dummy_category", expected_class=plugins.Plugin)
    init_file = os.path.join(str(test_plugins_dir), "__init__.py")
    with pytest.raises(RuntimeError, match=".*[Ii]mport failed.*"):
        plugins.importing.import_plugins(init_file)
