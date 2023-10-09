import tempor.core.plugins as plugins

from .._base import BaseScaler

plugins.register_plugin_category("preprocessing.scaling.temporal", BaseScaler)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
