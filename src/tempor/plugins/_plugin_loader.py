import tempor.plugins.core as plugins

from . import classification, preprocessing, regression, time_to_event, treatments

plugin_loader = plugins.PluginLoader()


__all__ = [
    "classification",
    "preprocessing",
    "regression",
    "time_to_event",
    "treatments",
]
