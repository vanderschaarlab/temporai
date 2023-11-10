"""Encoding methods for temporal data."""

from tempor.core import plugins

from .._base import BaseEncoder

plugins.register_plugin_category("preprocessing.encoding.temporal", BaseEncoder)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
