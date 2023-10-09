from typing import Dict, Type

import pytest

import tempor.methods.core._plugin as plugin_core

DUMMY_PLUGIN_CATEGORY_REGISTRY: Dict[str, Type[plugin_core.Plugin]] = dict()
DUMMY_PLUGIN_REGISTRY: Dict[str, Type[plugin_core.Plugin]] = dict()


@pytest.fixture
def patch_plugins_core_module(patch_module):
    patch_module(
        main_module=plugin_core,
        module_vars=[
            (
                plugin_core,
                plugin_core.PLUGIN_CATEGORY_REGISTRY,
                "PLUGIN_CATEGORY_REGISTRY",
                DUMMY_PLUGIN_CATEGORY_REGISTRY,
            ),
            (
                plugin_core,
                plugin_core.PLUGIN_REGISTRY,
                "PLUGIN_REGISTRY",
                DUMMY_PLUGIN_REGISTRY,
            ),
        ],
        refresh_pydantic=False,
    )
