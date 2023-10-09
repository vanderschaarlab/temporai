import tempor.core.plugins as plugins

from .._base import BaseImputer

plugins.register_plugin_category("preprocessing.imputation.static", BaseImputer)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
