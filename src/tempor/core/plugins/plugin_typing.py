from typing import Union

from typing_extensions import Literal, get_args

# Type aliases:
PluginType = str
"""Type alias to indicate plugin type, such as ``'method'``."""
PluginTypeArgAll = Literal["all"]
"""Literal for argument options indicating all plugin types."""
PluginTypeArg = Union[None, PluginTypeArgAll, str]
"""Plugin type argument type. May be `PluginType` (`str`), None, or `PluginTypeArgAll` (``"all"``)"""
PluginName = str
"""Type alias to indicate plugin name, such as ``'my_nn_classifier'``."""
PluginFullName = str
"""Type alias to indicate plugin full name (with all [sub]categories),
such as ``'prediction.one_off.classification.my_nn_classifier'``.
"""
PluginCategory = str
"""Type alias to indicate plugin category (including all [sub]categories),
such as ``'prediction.one_off.classification'``.
"""

# Internal type aliases:
_PluginFqn = str
"""Type alias to indicate plugin FQN, including both [plugin_type] and category,
such as ``'[method].prediction.one_off.classification.my_nn_classifier'``.
"""
_PluginCategoryFqn = str
"""Type alias to indicate plugin category FQN, including both [plugin_type] and category,
such as ``'[method].prediction.one_off.classification'``.
"""

# Default plugin type:
DEFAULT_PLUGIN_TYPE = "method"
"""Default plugin type to which plugins will be assigned if no plugin type is specified
(plugin type set to ``None``).
"""

# All plugin types indicator:
ALL_PLUGIN_TYPES_INDICATOR = get_args(PluginTypeArgAll)[0]
"""A string that indicates all plugins."""
