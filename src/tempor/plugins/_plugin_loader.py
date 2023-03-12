import tempor.plugins.core as plugins

from . import preprocessing, survival

plugin_loader = plugins.PluginLoader()


__all__ = [
    "preprocessing",
    "survival",
]
