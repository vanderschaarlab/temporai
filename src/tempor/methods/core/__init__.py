"""Package directory for the core functionality for the methods, mainly the base classes."""

from tempor.core.plugins import Plugin, PluginLoader, importing, register_plugin, register_plugin_category

from ._base_estimator import BaseEstimator
from ._base_predictor import BasePredictor
from ._base_transformer import BaseTransformer

__all__ = [
    "BaseEstimator",
    "importing",
    "Plugin",
    "PluginLoader",
    "BasePredictor",
    "register_plugin_category",
    "register_plugin",
    "BaseTransformer",
]
