import tempor.plugins.core as plugins

from . import classification, preprocessing, regression, tte

plugin_loader = plugins.PluginLoader()


__all__ = [
    "preprocessing",
    "tte",
    "classification",
    "regression",
]
