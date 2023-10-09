from typing import Dict, Type

import pytest

from tempor.core import plugins

DUMMY_PLUGIN_CATEGORY_REGISTRY: Dict[str, Type[plugins.Plugin]] = dict()
DUMMY_PLUGIN_REGISTRY: Dict[str, Type[plugins.Plugin]] = dict()


@pytest.fixture
def patch_plugins_core_module(patch_module):
    patch_module(
        main_module=plugins,
        module_vars=[
            (
                plugins,
                plugins.PLUGIN_CATEGORY_REGISTRY,
                "PLUGIN_CATEGORY_REGISTRY",
                DUMMY_PLUGIN_CATEGORY_REGISTRY,
            ),
            (
                plugins,
                plugins.PLUGIN_REGISTRY,
                "PLUGIN_REGISTRY",
                DUMMY_PLUGIN_REGISTRY,
            ),
        ],
        refresh_pydantic=False,
    )
