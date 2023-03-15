import tempor.plugins.core as plugins

from . import classification, preprocessing, regression, survival

plugin_loader = plugins.PluginLoader()


__all__ = [
    "preprocessing",
    "survival",
    "classification",
    "regression",
]
