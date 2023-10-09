import tempor.core.plugins as plugins

from .._base import BaseEncoder

plugins.register_plugin_category("preprocessing.encoding.static", BaseEncoder)

plugins.importing.import_plugins(__file__)

__all__ = [
    *plugins.importing.gather_modules_names(__file__),
]
