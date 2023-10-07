import glob
import importlib
import importlib.abc
import importlib.util
import os
import os.path
import sys
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Tuple, Type, TypeVar, Union, overload

from typing_extensions import Literal, ParamSpec, get_args

import tempor
from tempor.core.utils import get_from_args_or_kwargs
from tempor.log import logger

from . import utils

PLUGIN_NAME_NOT_SET = "NOT_SET"
PLUGIN_CATEGORY_NOT_SET = "NOT_SET"
PLUGIN_TYPE_NOT_SET = "NOT_SET"


P = ParamSpec("P")
T = TypeVar("T")

# Type aliases:
PluginTypeAll = Literal["all"]
"""Literal for argument options indicating all plugin types."""
PluginType = Union[None, PluginTypeAll, str]
"""Plugin type argument type."""
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


# Local helpers. ---


def parse_plugin_type(plugin_type: PluginType) -> PluginType:
    """Get the default plugin type if ``plugin_type`` is ``None``. If ``plugin_type`` is ``"all"``, raise error,
    as that is a reserved value.

    Args:
        plugin_type (PluginType): Plugin type.

    Returns:
        PluginType: Default plugin type if ``plugin_type`` is ``None``, otherwise ``plugin_type``.
    """
    if plugin_type == get_args(PluginTypeAll)[0]:
        raise ValueError(f"Plugin type cannot be '{plugin_type}' as that is a reserved value.")
    if plugin_type is None:
        return DEFAULT_PLUGIN_TYPE
    return plugin_type


def create_fqn(suffix: Union[PluginCategory, PluginFullName], plugin_type: PluginType) -> str:
    """Create a fully-qualified name for a plugin or category, like `[plugin_type].category.name` or
    `[plugin_type].category` respectively.

    Args:
        suffix (Union[PluginCategory, PluginFullName]): Plugin category or plugin full name.
        plugin_type (PluginType): Plugin type.

    Returns:
        str: Fully-qualified name.
    """
    if plugin_type is None:
        raise ValueError("Plugin type cannot be `None`. Did you forget to call `get_default_plugin_type`?")
    return f"[{plugin_type}].{suffix}"


def filter_list_by_plugin_type(lst: List[_PluginFqn], plugin_type: PluginType) -> List[PluginFullName]:
    """Filter a list of plugin FQNs by plugin type.

    Args:
        lst (List[_PluginFqn]): List of plugin FQNs.
        plugin_type (PluginType): Plugin type.

    Returns:
        List[PluginFullName]: Filtered list which will only include FQNs with the specified ``plugin_type``.
    """
    return [x for x in lst if x.split(".")[0] == f"[{plugin_type}]"]


def filter_dict_by_plugin_type(d: Dict[_PluginFqn, Any], plugin_type: PluginType) -> Dict[PluginFullName, Any]:
    """Filter a dictionary with plugin FQN keys by plugin type.

    Args:
        d (Dict[_PluginFqn, Any]): Dictionary to filter.
        plugin_type (PluginType): Plugin type.

    Returns:
        Dict[PluginFullName, Any]: Filtered dictionary which will only include items where FQN keys match the \
            specified ``plugin_type``.
    """
    return {k: v for k, v in d.items() if k.split(".")[0] == f"[{plugin_type}]"}


def _parse_fqn_format(fqn: str) -> Tuple[PluginType, str]:
    """Parse a plugin FQN or category FQN into its plugin type and remainder (``category`` or ``category.name``) parts.

    Args:
        fqn (str): Plugin FQN or category FQN.

    Raises:
        ValueError: Raised if the FQN is of incorrect format, that is, doesn't begin with ``[<plugin_type>].<...>``.

    Returns:
        Tuple[PluginType, str]: Plugin type, remainder (``category`` or ``category.name``).
    """
    first_element = fqn.split(".")[0]
    if not (first_element[0] == "[" and first_element[-1] == "]"):
        raise ValueError(f"FQN '{fqn}' is of incorrect format, expected to begin with `[<plugin_type>].<...>`")
    plugin_type = first_element[1:-1]
    remainder = ".".join(fqn.split(".")[1:])
    return plugin_type, remainder


def remove_plugin_type_from_fqn(fqn: Union[_PluginCategoryFqn, _PluginFqn]) -> Union[PluginCategory, PluginFullName]:
    """Remove the plugin type part of a plugin FQN or category FQN.

    Args:
        fqn (Union[_PluginCategoryFqn, _PluginFqn]): Plugin FQN of plugin category FQN.

    Returns:
        str: The FQN with the plugin type part removed.
    """
    _, remainder = _parse_fqn_format(fqn)
    return remainder


def get_plugin_type_from_fqn(fqn: Union[_PluginCategoryFqn, _PluginFqn]) -> PluginType:
    """Get the plugin type part of a plugin FQN or category FQN.

    Args:
        fqn (Union[_PluginCategoryFqn, _PluginFqn]): Plugin FQN of plugin category FQN.

    Returns:
        PluginType: The plugin type.
    """
    plugin_type, _ = _parse_fqn_format(fqn)
    return plugin_type


# Local helpers (end). ---


class Plugin:
    """The base class that all plugins must inherit from."""

    name: ClassVar[PluginName] = PLUGIN_NAME_NOT_SET
    """Plugin name, such as ``'my_nn_classifier'``. Must be set by the plugin class using ``@register_plugin``."""
    category: ClassVar[PluginCategory] = PLUGIN_CATEGORY_NOT_SET
    """Plugin category, such as ``'prediction.one_off.classification'``.
    Must be set by the plugin class using ``@register_plugin``.
    """
    plugin_type: ClassVar[PluginType] = PLUGIN_TYPE_NOT_SET
    """Plugin type, such as ``'method'``. May be optionally set by the plugin class using ``@register_plugin``,
    else will set the default plugin type.
    """

    @classmethod
    def full_name(cls) -> str:
        """The full name of the plugin with its category: ``category.subcategory.name``.
        There may be 0 or more subcategories.
        """
        return f"{cls.category}.{cls.name}"

    @classmethod
    def _fqn(cls) -> _PluginFqn:
        """The fully-qualified name of the plugin with its plugin type: ``[plugin_type].category.subcategory.name``"""
        return f"{create_fqn(cls.category, cls.plugin_type)}.{cls.name}"

    @classmethod
    def _category_fqn(cls) -> _PluginCategoryFqn:
        """The fully-qualified name of the plugin's category: ``[plugin_type].category.subcategory``"""
        return f"{create_fqn(cls.category, cls.plugin_type)}"

    def __init__(self) -> None:
        if self.name == PLUGIN_NAME_NOT_SET:
            raise ValueError(f"Plugin {self.__class__.__name__} `name` was not set, use @{register_plugin.__name__}")
        if self.category == PLUGIN_CATEGORY_NOT_SET:
            raise ValueError(
                f"Plugin {self.__class__.__name__} `category` was not set, use @{register_plugin.__name__}"
            )
        if self.plugin_type == PLUGIN_TYPE_NOT_SET:
            raise ValueError(
                f"Plugin {self.__class__.__name__} `plugin_type` was not set, use @{register_plugin.__name__}"
            )


# Important dicts that store plugin information:
PLUGIN_CATEGORY_REGISTRY: Dict[_PluginCategoryFqn, Type[Plugin]] = dict()
"""Important dictionary for plugin functionality. Records all plugin categories
(``'[plugin_type].category.<0 or more subcategories if applicable>'``) and their corresponding plugin classes."""
PLUGIN_REGISTRY: Dict[_PluginFqn, Type[Plugin]] = dict()
"""Important dictionary for plugin functionality. Records all plugins by their fully-qualified name
``'[plugin_type].category.<0 or more subcategories if applicable>.plugin_name'``."""


def register_plugin_category(category: PluginCategory, expected_class: Type, plugin_type: PluginType = None) -> None:
    """A decorator to register a plugin category (with optional subcategories). If ``plugin_type`` is provided,
    this will also be assigned, otherwise the default plugin type will be used.

    Args:
        category (PluginCategory): Plugin category, dot-separated, with optional subcategories, \
            such as ``'prediction.one_off.classification'``.
        expected_class (Type): The expected plugin class for this category. The plugin class must be a subclass of \
            this class. Note that this class must itself be a subclass of ``Plugin``.
        plugin_type (PluginType, optional): Plugin type to register the category under. Different plugin types should \
            be used to indicate different domains of your code (e.g. methods vs metrics vs datasets). \
            Defaults to `None`.

    Raises:
        TypeError: If the ``expected_class`` is not correctly defined.
    """
    plugin_type = parse_plugin_type(plugin_type)
    logger.debug(f"Registering plugin type {plugin_type}")
    logger.debug(f"Registering plugin category {category}")
    category_fqn = create_fqn(suffix=category, plugin_type=plugin_type)
    if category_fqn in PLUGIN_CATEGORY_REGISTRY:
        raise TypeError(f"Plugin category {category} already registered under plugin type {plugin_type}")
    if not issubclass(expected_class, Plugin):
        raise TypeError(f"Plugin expected class for category should be a subclass of {Plugin} but was {expected_class}")
    PLUGIN_CATEGORY_REGISTRY[category_fqn] = expected_class


def _check_same_class(class_1, class_2) -> bool:
    # To avoid raising "already registered" error when a certain plugin class is being reimported.
    # Not a bullet proof check but should suffice for practical purposes.
    return (
        class_1.__name__ == class_2.__name__ and class_1.__module__.split(".")[-1] == class_2.__module__.split(".")[-1]
    )


def register_plugin(name: str, category: PluginCategory, plugin_type: PluginType = None):
    """A decorator to register a plugin class. If ``plugin_type`` is provided, this will also be assigned,
    otherwise the default plugin type will be used. The ``category`` must have already been registered with
    ``@register_plugin_category`` before this can be used to register a plugin.

    Args:
        name (str): Plugin name, such as ``'my_nn_classifier'``.
        category (PluginCategory): Plugin category, dot-separated, with optional subcategories, \
            such as ``'prediction.one_off.classification'``.
        plugin_type (PluginType, optional): Plugin type of the category. If left as `None`, default plugin type is \
            assumed. ``plugin_type`` must correctly correspond to the ``category`` being specified. \
            Defaults to `None`.
    """

    def _class_decorator(cls: Callable[P, T]) -> Callable[P, T]:
        # NOTE:
        # The Callable[<ParamSpec>, <TypeVar>] approach allows to preserve the type annotation of the parameters of the
        # wrapped class (its __init__ method, specifically). See resources:
        #     * https://stackoverflow.com/a/74080156
        #     * https://docs.python.org/3/library/typing.html#typing.ParamSpec

        if TYPE_CHECKING:  # pragma: no cover
            # Note that cls is in fact `Type[Plugin]`, but this allows to
            # silence static type-checker warnings inside this function.
            assert isinstance(cls, Plugin)  # nosec B101

        logger.debug(f"Registering plugin of class {cls}")
        cls.name = name
        cls.category = category

        _plugin_type = parse_plugin_type(plugin_type)
        cls.plugin_type = _plugin_type

        category_fqn = create_fqn(suffix=category, plugin_type=_plugin_type)

        if category_fqn not in PLUGIN_CATEGORY_REGISTRY:
            raise TypeError(
                f"Found plugin category '{cls.category}' under plugin type '{cls.plugin_type}' which "
                f"has not been registered with `@{register_plugin_category.__name__}`"
            )
        if not issubclass(cls, Plugin):
            raise TypeError(f"Expected plugin class `{cls.__name__}` to be a subclass of `{Plugin}` but was `{cls}`")
        if not issubclass(cls, PLUGIN_CATEGORY_REGISTRY[category_fqn]):
            raise TypeError(
                f"Expected plugin class `{cls.__name__}` to be a subclass of "
                f"`{PLUGIN_CATEGORY_REGISTRY[category_fqn]}` but was `{cls}`"
            )
        # pylint: disable-next=protected-access
        if cls._fqn() in PLUGIN_REGISTRY:
            # pylint: disable-next=protected-access
            if not _check_same_class(cls, PLUGIN_REGISTRY[cls._fqn()]):
                raise TypeError(
                    # pylint: disable-next=protected-access
                    f"Plugin (plugin type '{cls.plugin_type}') with full name '{cls.full_name()}' has already been "
                    f"registered (as class `{PLUGIN_REGISTRY[cls._fqn()]}`)"
                )
            else:
                # The same class is being reimported, do not raise error.
                pass
        for existing_cls in PLUGIN_REGISTRY.values():
            # Cannot have the same plugin name (not just fqn), as this is not supported by Pipeline.
            # TODO: Fix this - make non-unique name work with pipeline.
            if cls.name == existing_cls.name:
                if not _check_same_class(cls, existing_cls):
                    raise TypeError(
                        f"Plugin (plugin type '{cls.plugin_type}') with name '{cls.name}' has already been "
                        f"registered (as class `{existing_cls}`). Must use a unique plugin name."
                    )
                else:  # pragma: no cover
                    # The same class is being reimported, do not raise error.
                    # Some kind of coverage issue - this case *is* covered by test:
                    # test_plugins.py::TestRegistration::test_category_registration_reimport_allowed
                    pass

        # pylint: disable-next=protected-access
        PLUGIN_REGISTRY[cls._fqn()] = cls

        return cls

    return _class_decorator


# TODO: Add "list all" option, perhaps when "None" is passed in to plugin_type, in all the relevant listing methods.
# TODO: Add "list types".
# TODO: Add check plugin type exists before listing.
# TODO: Consider whether to enforce common base class across plugin_type/category.
class PluginLoader:
    """A class to load plugins. Provides functionality to list and get plugins."""

    def __init__(self) -> None:
        self._refresh()

    def _refresh(self):
        # Internal method to refresh plugin registries.

        self._plugin_registry: Dict[str, Type[Plugin]] = PLUGIN_REGISTRY

        name_by_category_nested: Dict = dict()
        for plugin_class in self._plugin_registry.values():
            name_by_category_nested = utils.append_by_dot_path(
                name_by_category_nested,
                key_path=plugin_class._category_fqn(),  # pylint: disable=protected-access
                value=plugin_class.name,
            )
        self._plugin_name_by_category_nested = name_by_category_nested

        class_by_category_nested: Dict = dict()
        for plugin_class in self._plugin_registry.values():
            class_by_category_nested = utils.append_by_dot_path(
                class_by_category_nested,
                key_path=plugin_class._category_fqn(),  # pylint: disable=protected-access
                value=plugin_class,
            )
        self._plugin_class_by_category_nested = class_by_category_nested

    def list(self, plugin_type: PluginType = None) -> Dict:
        """List all plugins of ``plugin_type`` as a nested dictionary, where the keys are the plugin categories
        and optional subcategories. The values of the dictionary are the plugin names.

        Args:
            plugin_type (PluginType, optional): Plugin type for which to list. Use default category if `None`. \
                Defaults to `None`.

        Returns:
            Dict: A dictionary as described above.
        """
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type)
        return self._plugin_name_by_category_nested[f"[{plugin_type}]"]

    def list_full_names(self, plugin_type: PluginType = None) -> List[PluginFullName]:
        """List all plugins of ``plugin_type`` as a list of plugin full names (including categories).

        Args:
            plugin_type (PluginType, optional): Plugin type for which to list. Use default category if `None`. \
                Defaults to `None`.

        Returns:
            List[PluginFullName]: A list as described above.
        """
        self._refresh()
        plugin_fqns = list(self._plugin_registry.keys())
        plugin_type = parse_plugin_type(plugin_type)
        plugin_fqns_filtered_by_type = filter_list_by_plugin_type(lst=plugin_fqns, plugin_type=plugin_type)
        return [remove_plugin_type_from_fqn(n) for n in plugin_fqns_filtered_by_type]

    def list_classes(self, plugin_type: PluginType = None) -> Dict:
        """List all plugin classes of ``plugin_type`` as a nested dictionary, where the keys are the plugin categories
        and optional subcategories. The values of the dictionary are the plugin **classes**.

        Args:
            plugin_type (PluginType, optional): Plugin type for which to list. Use default category if `None`. \
                Defaults to `None`.

        Returns:
            Dict: A dictionary as described above.
        """
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type)
        return self._plugin_class_by_category_nested[f"[{plugin_type}]"]

    def list_categories(self, plugin_type: PluginType = None) -> Dict[PluginFullName, Type[Plugin]]:
        """List all plugin categories of ``plugin_type`` as a dictionary, where the keys are the plugin category names
        (including optional subcategories) and the values are the **expected plugin classes** for that category.

        Args:
            plugin_type (PluginType, optional): Plugin type for which to list. Use default category if `None`. \
                Defaults to `None`.

        Returns:
            Dict[PluginFullName, Type[Plugin]]: A dictionary as described above.
        """
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type)
        categories_filtered_by_type = filter_dict_by_plugin_type(d=PLUGIN_CATEGORY_REGISTRY, plugin_type=plugin_type)
        return {remove_plugin_type_from_fqn(k): v for k, v in categories_filtered_by_type.items()}

    def _raise_plugin_does_not_exist_error(self, fqn: str):
        plugin_type = get_plugin_type_from_fqn(fqn)
        plugin_full_name = remove_plugin_type_from_fqn(fqn)
        if fqn not in self._plugin_registry:
            raise ValueError(f"Plugin '{plugin_full_name}' (plugin type: {plugin_type}) does not exist.")

    def _handle_get_args_kwargs(self, args, kwargs) -> Tuple[Any, Tuple, Dict]:
        # "Pop" the `plugin_type` argument if such is found in args (position 0) or kwargs.
        # If appears to be provided in both ways, prefer the value from kwargs and leave the string in args as is.
        # If not, `plugin_type` will fall back to its default value.
        plugin_type, args, kwargs = get_from_args_or_kwargs(
            args, kwargs, argument_name="plugin_type", argument_type=str, position_if_args=0, prefer="kwarg"
        )
        if plugin_type is None:
            plugin_type = parse_plugin_type(plugin_type)
        return plugin_type, args, kwargs

    @overload
    def get(self, name: PluginFullName, plugin_type: PluginType, *args, **kwargs) -> Type:
        ...

    @overload
    def get(self, name: PluginFullName, *args, plugin_type: PluginType = None, **kwargs) -> Type:
        ...

    def get(self, name: PluginFullName, *args, **kwargs) -> Any:
        """Get a plugin by its full name (including category, i.e. of form
        ``'my_category.my_subcategory.my_plugin'``). Use ``*args`` and ``**kwargs`` to pass arguments to
        the plugin initializer. The returned object is an instance of the plugin class. If the plugin is not of the
        default plugin type, must provide ``plugin_type`` also.

        The method can be called with ``plugin_type`` and plugin initializer arguments, as follows:

        - As first positional argument after the plugin name:
            >>> plugin_instance = get(  # doctest: +SKIP
            ...     "my_category.my_subcategory.my_plugin",  # Plugin full name.
            ...     "method",  # Plugin type provided as a positional argument (first).
            ...     0.4,  # First positional argument to plugin initializer.
            ...     123,  # Second positional argument to plugin initializer...
            ...     kwarg=2,  # Keyword argument(s) to plugin initializer from here on.
            ... )

        - As keyword argument:
            >>> plugin_instance = get(  # doctest: +SKIP
            ...     "my_category.my_subcategory.my_plugin",  # Plugin full name.
            ...     0.4,  # First positional argument to plugin initializer.
            ...     123,  # Second positional argument to plugin initializer...
            ...     plugin_type="method",  # Plugin type provided as a keyword argument.
            ...     kwarg=2,  # Keyword argument(s) to plugin initializer from here on.
            ... )

        Args:
            name (PluginFullName): Plugin full name including all (sub)categories, of form \
                ``'my_category.my_subcategory.my_plugin'``

        Returns:
            Any: The plugin instance initialised with ``*args`` and ``**kwargs`` as provided.
        """
        plugin_type, args, kwargs = self._handle_get_args_kwargs(args, kwargs)
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type)
        fqn = create_fqn(suffix=name, plugin_type=plugin_type)
        self._raise_plugin_does_not_exist_error(fqn)
        return self._plugin_registry[fqn](*args, **kwargs)

    def get_class(self, name: PluginFullName, plugin_type: PluginType = None) -> Type:
        """Get a plugin class (not instance) by its full name (including category, i.e. of form
        ``'my_category.my_subcategory.my_plugin'``). If the plugin is not of the default plugin type, must provide
        ``plugin_type``.

        Args:
            name (PluginFullName): Plugin full name including all (sub)categories, of form \
                ``'my_category.my_subcategory.my_plugin'``
            plugin_type (PluginType, optional): Plugin type. If left as `None`, default plugin type is assumed. \
                ``plugin_type`` must correctly correspond to the category implied by plugin full name. \
                Defaults to `None`.

        Returns:
            Type: Plugin class (not instance).
        """
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type)
        fqn = create_fqn(suffix=name, plugin_type=plugin_type)
        self._raise_plugin_does_not_exist_error(fqn)
        return self._plugin_registry[fqn]


PLUGIN_FILENAME_PREFIX = "plugin_"
"""Prefix expected for plugin filenames of python files that contain plugin code."""


def _glob_plugin_paths(package_dir: str) -> List[str]:
    # Get the paths of all python files in the package directory that begin with `PLUGIN_FILENAME_PREFIX`.
    return [f for f in glob.glob(os.path.join(package_dir, f"{PLUGIN_FILENAME_PREFIX}*.py")) if os.path.isfile(f)]


def _module_name_from_path(path: str) -> str:
    # Return a module name from a path to a python file, raise exception if not a python file.
    path = os.path.normpath(path)
    split = path[path.rfind(f"{tempor.import_name}{os.sep}") :].split(os.sep)
    if split[-1][-3:] != ".py":  # pragma: no cover
        # Should be prevented by `_glob_plugin_paths`.
        raise ValueError(f"Path `{path}` is not a python file.")
    split[-1] = split[-1].replace(".py", "")
    return ".".join(split)


class importing:
    """A namespace for plugin importing functionality."""

    @staticmethod
    def import_plugins(init_file: str) -> None:
        """Import all plugin modules inside the package directory associated with ``init_file`` (``__init__.py``).
        Importing in this context means programmatic import and execution of the plugin modules.

        Args:
            init_file (str): The init file for the package directory containing the plugin modules (files).

        Raises:
            RuntimeError: Raised if there are import problems with any of the plugin modules.
        """
        package_dir = os.path.dirname(init_file)
        logger.debug(f"Importing all plugin modules inside {package_dir}")
        paths = _glob_plugin_paths(package_dir=package_dir)
        logger.trace(f"Found plugin module paths to import:\n{paths}")
        for f in paths:
            module_name = _module_name_from_path(f)
            logger.debug(f"Importing plugin module: {module_name}")
            spec = importlib.util.spec_from_file_location(module_name, f)
            if spec is None or not isinstance(spec.loader, importlib.abc.Loader):
                raise RuntimeError(f"Import failed for {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    @staticmethod
    def gather_modules_names(package_init_file: str) -> List[str]:
        """Gather the names of all plugin modules inside the package directory associated with ``init_file``.
        Useful for e.g. setting the ``__all__`` variable.

        Args:
            package_init_file (str): The init file for the package directory containing the plugin modules (files).

        Returns:
            List[str]: A list of plugin module names.
        """
        package_dir = os.path.dirname(package_init_file)
        paths = _glob_plugin_paths(package_dir=package_dir)
        return [_module_name_from_path(f) for f in paths]
