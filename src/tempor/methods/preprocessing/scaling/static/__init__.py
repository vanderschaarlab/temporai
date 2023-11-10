"""Scaling methods for static data."""

from tempor.core import plugins

from .._base import BaseScaler

plugins.register_plugin_category("preprocessing.scaling.static", BaseScaler)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
