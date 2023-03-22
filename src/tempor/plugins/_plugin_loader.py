import tempor.plugins.core as plugins

from . import classification, preprocessing, regression, time_to_event

plugin_loader = plugins.PluginLoader()


__all__ = [
    "preprocessing",
    "time_to_event",
    "classification",
    "regression",
]
