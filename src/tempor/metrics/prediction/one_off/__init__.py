"""Metrics for the one-off prediction tasks."""

from tempor.core import plugins

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
