import tempor.methods.core as plugins

from . import prediction, preprocessing, time_to_event, treatments

plugin_loader = plugins.PluginLoader()


__all__ = [
    "prediction",
    "preprocessing",
    "time_to_event",
    "treatments",
]
