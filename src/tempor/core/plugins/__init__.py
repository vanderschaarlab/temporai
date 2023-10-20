import glob
import importlib
import importlib.abc
import importlib.util
import os
import os.path
import sys
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast, overload

from typing_extensions import ParamSpec

import tempor
from tempor.log import logger

from .. import utils as core_utils
from . import _utils as plugin_utils
from . import plugin_typing

PLUGIN_NAME_NOT_SET = "NOT_SET"
PLUGIN_CATEGORY_NOT_SET = "NOT_SET"
PLUGIN_TYPE_NOT_SET = "NOT_SET"


P = ParamSpec("P")
T = TypeVar("T")


# Local helpers. ---


def parse_plugin_type(
    plugin_type: plugin_typing.PluginTypeArg, must_be_one_of: Optional[List[str]] = None
) -> plugin_typing.PluginTypeArg:
    """Get the default plugin type if ``plugin_type`` is ``None``. If ``plugin_type`` is ``"all"``, raise error,
    as that is a reserved value.

    Args:
        plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg):
            Plugin type.
        must_be_one_of (List[str]):
            List of plugin types that ``plugin_type`` must be one of.

    Returns:
        ~tempor.core.plugin.plugin_typing.PluginTypeArg:
            Default plugin type if ``plugin_type`` is ``None``, otherwise ``plugin_type``.
    """
    if plugin_type == plugin_typing.ALL_PLUGIN_TYPES_INDICATOR:
        raise ValueError(f"Plugin type cannot be '{plugin_type}' as that is a reserved value.")
    if plugin_type is None:
        return plugin_typing.DEFAULT_PLUGIN_TYPE
    if must_be_one_of and plugin_type not in must_be_one_of:
        raise ValueError(f"Plugin type must be one of {must_be_one_of} but was '{plugin_type}'")
    return plugin_type


def create_fqn(
    suffix: Union[plugin_typing.PluginCategory, plugin_typing.PluginFullName], plugin_type: plugin_typing.PluginTypeArg
) -> str:
    """Create a fully-qualified name for a plugin or category, like `[plugin_type].category.name` or
    `[plugin_type].category` respectively.

    Args:
        suffix (Union[~tempor.core.plugin.plugin_typing.PluginCategory, \
            ~tempor.core.plugin.plugin_typing.PluginFullName]):
            Plugin category or plugin full name.
        plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg):
            Plugin type.

    Returns:
        str: Fully-qualified name.
    """
    if plugin_type is None:
        raise ValueError("Plugin type cannot be `None`. Did you forget to call `get_default_plugin_type`?")
    return f"[{plugin_type}].{suffix}"


def filter_list_by_plugin_type(
    lst: List[plugin_typing._PluginFqn], plugin_type: plugin_typing.PluginTypeArg
) -> List[plugin_typing.PluginFullName]:
    """Filter a list of plugin FQNs by plugin type.

    Args:
        lst (List[~tempor.core.plugin.plugin_typing._PluginFqn]):
            List of plugin FQNs.
        plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg):
            Plugin type.

    Returns:
        List[~tempor.core.plugin.plugin_typing.PluginFullName]:
            Filtered list which will only include FQNs with the specified ``plugin_type``.
    """
    return [x for x in lst if x.split(".")[0] == f"[{plugin_type}]"]


def filter_dict_by_plugin_type(
    d: Dict[plugin_typing._PluginFqn, Any], plugin_type: plugin_typing.PluginTypeArg
) -> Dict[plugin_typing.PluginFullName, Any]:
    """Filter a dictionary with plugin FQN keys by plugin type.

    Args:
        d (Dict[~tempor.core.plugin.plugin_typing._PluginFqn, Any]):
            Dictionary to filter.
        plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg):
            Plugin type.

    Returns:
        Dict[~tempor.core.plugin.plugin_typing.PluginFullName, Any]:
            Filtered dictionary which will only include items where FQN keys match the specified ``plugin_type``.
    """
    return {k: v for k, v in d.items() if k.split(".")[0] == f"[{plugin_type}]"}


def _parse_fqn_format(fqn: str) -> Tuple[plugin_typing.PluginTypeArg, str]:
    """Parse a plugin FQN or category FQN into its plugin type and remainder (``category`` or ``category.name``) parts.

    Args:
        fqn (str):
            Plugin FQN or category FQN.

    Raises:
        ValueError:
            Raised if the FQN is of incorrect format, that is, doesn't begin with ``[<plugin_type>].<...>``.

    Returns:
        Tuple[~tempor.core.plugin.plugin_typing.PluginTypeArg, str]:
            Plugin type, remainder (``category`` or ``category.name``).
    """
    first_element = fqn.split(".")[0]
    if not (first_element[0] == "[" and first_element[-1] == "]"):
        raise ValueError(f"FQN '{fqn}' is of incorrect format, expected to begin with `[<plugin_type>].<...>`")
    plugin_type = first_element[1:-1]
    remainder = ".".join(fqn.split(".")[1:])
    return plugin_type, remainder


def remove_plugin_type_from_fqn(
    fqn: Union[plugin_typing._PluginCategoryFqn, plugin_typing._PluginFqn]
) -> Union[plugin_typing.PluginCategory, plugin_typing.PluginFullName]:
    """Remove the plugin type part of a plugin FQN or category FQN.

    Args:
        fqn (Union[~tempor.core.plugin.plugin_typing._PluginCategoryFqn, ~tempor.core.plugin.plugin_typing._PluginFqn]):
            Plugin FQN of plugin category FQN.

    Returns:
        Union[~tempor.core.plugin.plugin_typing.PluginCategory, ~tempor.core.plugin.plugin_typing.PluginFullName]:
            The FQN with the plugin type part removed.
    """
    _, remainder = _parse_fqn_format(fqn)
    return remainder


def get_plugin_type_from_fqn(
    fqn: Union[plugin_typing._PluginCategoryFqn, plugin_typing._PluginFqn]
) -> plugin_typing.PluginTypeArg:
    """Get the plugin type part of a plugin FQN or category FQN.

    Args:
        fqn (Union[~tempor.core.plugin.plugin_typing._PluginCategoryFqn, ~tempor.core.plugin.plugin_typing._PluginFqn]):
            Plugin FQN of plugin category FQN.

    Returns:
        ~tempor.core.plugin.plugin_typing.PluginTypeArg: The plugin type.
    """
    plugin_type, _ = _parse_fqn_format(fqn)
    return plugin_type


# Local helpers (end). ---


class Plugin:
    """The base class that all plugins must inherit from."""

    name: ClassVar[plugin_typing.PluginName] = PLUGIN_NAME_NOT_SET
    """Plugin name, such as ``'my_nn_classifier'``. Must be set by the plugin class using ``@register_plugin``."""
    category: ClassVar[plugin_typing.PluginCategory] = PLUGIN_CATEGORY_NOT_SET
    """Plugin category, such as ``'prediction.one_off.classification'``.
    Must be set by the plugin class using ``@register_plugin``.
    """
    plugin_type: ClassVar[plugin_typing.PluginTypeArg] = PLUGIN_TYPE_NOT_SET
    """Plugin type, such as ``'method'``. May be optionally set by the plugin class using ``@register_plugin``,
    else will set the default plugin type.
    """

    @classmethod
    def full_name(cls) -> plugin_typing.PluginFullName:
        """The full name of the plugin with its category: ``category.subcategory.name``.
        There may be 0 or more subcategories.

        Returns:
            ~tempor.core.plugin.plugin_typing.PluginFullName: Plugin full name.
        """
        return f"{cls.category}.{cls.name}"

    @classmethod
    def _fqn(cls) -> plugin_typing._PluginFqn:
        """The fully-qualified name of the plugin with its plugin type: ``[plugin_type].category.subcategory.name``

        Returns:
            ~tempor.core.plugin.plugin_typing._PluginFqn: Plugin fully-qualified name.
        """
        return f"{create_fqn(cls.category, cls.plugin_type)}.{cls.name}"

    @classmethod
    def _category_fqn(cls) -> plugin_typing._PluginCategoryFqn:
        """The fully-qualified name of the plugin's category: ``[plugin_type].category.subcategory``

        Returns:
            ~tempor.core.plugin.plugin_typing._PluginCategoryFqn: Plugin category fully-qualified name.
        """
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
PLUGIN_CATEGORY_REGISTRY: Dict[plugin_typing._PluginCategoryFqn, Type[Plugin]] = dict()
"""Important dictionary for plugin functionality. Records all plugin categories
(``'[plugin_type].category.<0 or more subcategories if applicable>'``) and their corresponding plugin classes."""
PLUGIN_REGISTRY: Dict[plugin_typing._PluginFqn, Type[Plugin]] = dict()
"""Important dictionary for plugin functionality. Records all plugins by their fully-qualified name
``'[plugin_type].category.<0 or more subcategories if applicable>.plugin_name'``."""


def register_plugin_category(
    category: plugin_typing.PluginCategory, expected_class: Type, plugin_type: plugin_typing.PluginTypeArg = None
) -> None:
    """A decorator to register a plugin category (with optional subcategories). If ``plugin_type`` is provided,
    this will also be assigned (or created, if such plugin type doesn't yet exist), otherwise the default plugin type
    will be used.

    Args:
        category (~tempor.core.plugin.plugin_typing.PluginCategory):
            Plugin category, dot-separated, with optional subcategories, \
            such as ``'prediction.one_off.classification'``.
        expected_class (Type):
            The expected plugin class for this category. The plugin class must be a subclass of \
            this class. Note that this class must itself be a subclass of ``Plugin``.
        plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg, optional):
            Plugin type to register the category under. Different plugin types should be used to indicate different \
            domains of your code (e.g. methods vs metrics vs datasets). Defaults to `None`.

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


def _check_same_class(class_1: Type, class_2: Type) -> bool:
    # To avoid raising "already registered" error when a certain plugin class is being reimported.
    # Not a bullet proof check but should suffice for practical purposes.
    return (
        class_1.__name__ == class_2.__name__ and class_1.__module__.split(".")[-1] == class_2.__module__.split(".")[-1]
    )


def register_plugin(
    name: str,
    category: plugin_typing.PluginCategory,
    plugin_type: plugin_typing.PluginTypeArg = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """A decorator to register a plugin class. If ``plugin_type`` is provided, this will also be assigned,
    otherwise the default plugin type will be used. The ``category`` must have already been registered with
    ``@register_plugin_category`` before this can be used to register a plugin.

    Args:
        name (str):
            Plugin name, such as ``'my_nn_classifier'``.
        category (~tempor.core.plugin.plugin_typing.PluginCategory):
            Plugin category, dot-separated, with optional subcategories, such as \
            ``'prediction.one_off.classification'``.
        plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg, optional):
            Plugin type of the category. If left as `None`, default plugin type is assumed. ``plugin_type`` must \
            correctly correspond to the ``category`` being specified. Defaults to `None`.
    """

    def _class_decorator(cls: Callable[P, T]) -> Callable[P, T]:
        # NOTE:
        # The Callable[<ParamSpec>, <TypeVar>] approach allows to preserve the type annotation of the parameters of the
        # wrapped class (its __init__ method, specifically). See resources:
        #     * https://stackoverflow.com/a/74080156
        #     * https://docs.python.org/3/library/typing.html#typing.ParamSpec

        # Cast to Type[Plugin], which is the actual expected type, such that static type checking works here.
        cls_ = cast(Type[Plugin], cls)

        logger.debug(f"Registering plugin of class {cls_}")
        cls_.name = name
        cls_.category = category

        _plugin_type = parse_plugin_type(plugin_type)
        cls_.plugin_type = _plugin_type

        category_fqn = create_fqn(suffix=category, plugin_type=_plugin_type)

        if category_fqn not in PLUGIN_CATEGORY_REGISTRY:
            raise TypeError(
                f"Found plugin category '{cls_.category}' under plugin type '{cls_.plugin_type}' which "
                f"has not been registered with `@{register_plugin_category.__name__}`"
            )
        if not issubclass(cls_, Plugin):
            raise TypeError(f"Expected plugin class `{cls_.__name__}` to be a subclass of `{Plugin}` but was `{cls_}`")
        if not issubclass(cls_, PLUGIN_CATEGORY_REGISTRY[category_fqn]):
            raise TypeError(
                f"Expected plugin class `{cls_.__name__}` to be a subclass of "
                f"`{PLUGIN_CATEGORY_REGISTRY[category_fqn]}` but was `{cls_}`"
            )
        # pylint: disable-next=protected-access
        if cls_._fqn() in PLUGIN_REGISTRY:
            # pylint: disable-next=protected-access
            if not _check_same_class(cls_, PLUGIN_REGISTRY[cls_._fqn()]):
                raise TypeError(
                    # pylint: disable-next=protected-access
                    f"Plugin (plugin type '{cls_.plugin_type}') with full name '{cls_.full_name()}' has already been "
                    f"registered (as class `{PLUGIN_REGISTRY[cls_._fqn()]}`)"
                )
            else:
                # The same class is being reimported, do not raise error.
                pass
        for existing_cls in PLUGIN_REGISTRY.values():
            # Cannot have the same plugin name (not just fqn), as this is not supported by Pipeline.
            # TODO: Fix this - make non-unique name work with pipeline.
            if cls_.name == existing_cls.name:
                if not _check_same_class(cls_, existing_cls):
                    raise TypeError(
                        f"Plugin (plugin type '{cls_.plugin_type}') with name '{cls_.name}' has already been "
                        f"registered (as class `{existing_cls}`). Must use a unique plugin name."
                    )
                else:  # pragma: no cover
                    # The same class is being reimported, do not raise error.
                    # Some kind of coverage issue - this case *is* covered by test:
                    # test_plugins.py::TestRegistration::test_category_registration_reimport_allowed
                    pass

        # pylint: disable-next=protected-access
        PLUGIN_REGISTRY[cls_._fqn()] = cls_

        # Cast back to Callable[P, T] (see note at the top of function).
        return cast(Callable[P, T], cls_)

    return _class_decorator


# TODO: Consider whether to enforce common base class across plugin_type/category.
class PluginLoader:
    """A class to load plugins. Provides functionality to list and get plugins."""

    def __init__(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        # Internal method to refresh plugin registries.

        self._plugin_registry: Dict[str, Type[Plugin]] = PLUGIN_REGISTRY

        name_by_category_nested: Dict = dict()
        for plugin_class in self._plugin_registry.values():
            name_by_category_nested = plugin_utils.append_by_dot_path(
                name_by_category_nested,
                key_path=plugin_class._category_fqn(),  # pylint: disable=protected-access
                value=plugin_class.name,
            )
        self._plugin_name_by_category_nested = name_by_category_nested

        class_by_category_nested: Dict = dict()
        for plugin_class in self._plugin_registry.values():
            class_by_category_nested = plugin_utils.append_by_dot_path(
                class_by_category_nested,
                key_path=plugin_class._category_fqn(),  # pylint: disable=protected-access
                value=plugin_class,
            )
        self._plugin_class_by_category_nested = class_by_category_nested

    def _handle_all_plugin_types_case(
        self,
        pt: plugin_typing.PluginTypeArg,
        method: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Dict:
        # If ``pt`` (plugin type) is "all", will call ``method`` for all plugin types and return a nested dictionary.
        # Otherwise, just calls ``method`` and return what it returns.
        # In either case, plugin type value(s) will be passed to ``method`` by ``plugin_type`` kwarg.
        if pt == plugin_typing.ALL_PLUGIN_TYPES_INDICATOR:
            output = dict()
            for actual_pt in self.list_plugin_types():
                output[actual_pt] = method(*args, **kwargs, plugin_type=actual_pt)
            return output
        else:
            return method(*args, **kwargs, plugin_type=pt)

    def _list(self, plugin_type: plugin_typing.PluginTypeArg = None) -> Dict:
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type, must_be_one_of=self.list_plugin_types())
        return self._plugin_name_by_category_nested[f"[{plugin_type}]"]

    def list(self, plugin_type: plugin_typing.PluginTypeArg = None) -> Dict:
        """List all plugins of ``plugin_type`` as a nested dictionary, where the keys are the plugin categories
        and optional subcategories. The values of the dictionary are the plugin names.

        If ``plugin_type`` is ``"all"``, will list for all plugin types, outputting inside a nested dictionary
        with plugin type keys.

        Args:
            plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg, optional):
                Plugin type for which to list. Use default category if `None`. Defaults to `None`.

        Returns:
            Dict: A dictionary as described above.
        """
        return self._handle_all_plugin_types_case(plugin_type, self._list)

    def _list_full_names(self, plugin_type: plugin_typing.PluginTypeArg = None) -> List[plugin_typing.PluginFullName]:
        self._refresh()
        plugin_fqns = list(self._plugin_registry.keys())
        plugin_type = parse_plugin_type(plugin_type, must_be_one_of=self.list_plugin_types())
        plugin_fqns_filtered_by_type = filter_list_by_plugin_type(lst=plugin_fqns, plugin_type=plugin_type)
        return [remove_plugin_type_from_fqn(n) for n in plugin_fqns_filtered_by_type]

    def list_full_names(
        self, plugin_type: plugin_typing.PluginTypeArg = None
    ) -> Union[List[plugin_typing.PluginFullName], Dict[str, List[plugin_typing.PluginFullName]]]:
        """List all plugins of ``plugin_type`` as a list of plugin full names (including categories).

        If ``plugin_type`` is ``"all"``, will list for all plugin types, outputting inside a nested dictionary
        with plugin type keys.

        Args:
            plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg, optional):
                Plugin type for which to list. Use default category if `None`. Defaults to `None`.

        Returns:
            Union[List[~tempor.core.plugin.plugin_typing.PluginFullName], \
            Dict[str, List[~tempor.core.plugin.plugin_typing.PluginFullName]]]:
                A list as described above (``List[PluginFullName]``) if ``plugin_type`` is not ``"all"``. \
                Otherwise a nested dictionary with plugin type keys and such lists as values \
                (``Dict[str, List[PluginFullName]]]``).
        """
        return self._handle_all_plugin_types_case(plugin_type, self._list_full_names)

    def _list_classes(self, plugin_type: plugin_typing.PluginTypeArg = None) -> Dict:
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type, must_be_one_of=self.list_plugin_types())
        return self._plugin_class_by_category_nested[f"[{plugin_type}]"]

    def list_classes(self, plugin_type: plugin_typing.PluginTypeArg = None) -> Dict:
        """List all plugin classes of ``plugin_type`` as a nested dictionary, where the keys are the plugin categories
        and optional subcategories. The values of the dictionary are the plugin **classes**.

        If ``plugin_type`` is ``"all"``, will list for all plugin types, outputting inside a nested dictionary
        with plugin type keys.

        Args:
            plugin_type (PluginType, optional):
                Plugin type for which to list. Use default category if `None`. Defaults to `None`.

        Returns:
            Dict: A dictionary as described above.
        """
        return self._handle_all_plugin_types_case(plugin_type, self._list_classes)

    def _list_categories(
        self, plugin_type: plugin_typing.PluginTypeArg = None
    ) -> Dict[plugin_typing.PluginFullName, Type[Plugin]]:
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type, must_be_one_of=self.list_plugin_types())
        categories_filtered_by_type = filter_dict_by_plugin_type(d=PLUGIN_CATEGORY_REGISTRY, plugin_type=plugin_type)
        return {remove_plugin_type_from_fqn(k): v for k, v in categories_filtered_by_type.items()}

    def list_categories(
        self, plugin_type: plugin_typing.PluginTypeArg = None
    ) -> Union[
        Dict[plugin_typing.PluginFullName, Type[Plugin]],
        Dict[plugin_typing.PluginType, Dict[plugin_typing.PluginFullName, Type[Plugin]]],
    ]:
        """List all plugin categories of ``plugin_type`` as a dictionary, where the keys are the plugin category names
        (including optional subcategories) and the values are the **expected plugin classes** for that category.

        If ``plugin_type`` is ``"all"``, will list for all plugin types, outputting inside a nested dictionary
        with plugin type keys.

        Args:
            plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg, optional):
                Plugin type for which to list. Use default category if `None`. Defaults to `None`.

        Returns:
            Union[Dict[~tempor.core.plugin.plugin_typing.PluginFullName, Type[Plugin]], \
            Dict[~tempor.core.plugin.plugin_typing.PluginType, \
            Dict[~tempor.core.plugin.plugin_typing.PluginFullName, Type[Plugin]]]]:
                A dictionary as described above (``Dict[PluginFullName, Type[Plugin]]``) if ``plugin_type`` is \
                not ``"all"``. Otherwise a nested dictionary with plugin type keys and such dictionaries as values \
                (``Dict[PluginType, Dict[PluginFullName, Type[Plugin]]]``).
        """
        return self._handle_all_plugin_types_case(plugin_type, self._list_categories)

    def list_plugin_types(self) -> List[plugin_typing.PluginType]:
        """List all plugin types.

        Returns:
            List[str]: A list of plugin types.
        """
        self._refresh()
        return core_utils.unique_in_order_of_appearance(
            [get_plugin_type_from_fqn(fqn) for fqn in self._plugin_registry.keys()]
        )

    def _raise_plugin_does_not_exist_error(self, fqn: str) -> None:
        plugin_type = get_plugin_type_from_fqn(fqn)
        plugin_full_name = remove_plugin_type_from_fqn(fqn)
        if fqn not in self._plugin_registry:
            raise ValueError(f"Plugin '{plugin_full_name}' (plugin type: {plugin_type}) does not exist.")

    def _handle_get_args_kwargs(self, args: Tuple, kwargs: Dict[str, Any]) -> Tuple[Any, Tuple, Dict]:
        # "Pop" the `plugin_type` argument if such is found in args (position 0) or kwargs.
        # If appears to be provided in both ways, prefer the value from kwargs and leave the string in args as is.
        # If not, `plugin_type` will fall back to its default value.
        plugin_type, args, kwargs = core_utils.get_from_args_or_kwargs(
            args, kwargs, argument_name="plugin_type", argument_type=str, position_if_args=0, prefer="kwarg"
        )
        if plugin_type is None:
            plugin_type = parse_plugin_type(None)
        return plugin_type, args, kwargs

    # Explicitly listing all the overloads for clarity of documentation.
    @overload
    def get(self, name: plugin_typing.PluginFullName, *args: Any, **kwargs: Any) -> Type:
        ...  # pragma: co cover

    @overload
    def get(  # type: ignore [misc]
        self,
        name: plugin_typing.PluginFullName,
        plugin_type: plugin_typing.PluginTypeArg,
        *args: Any,
        **kwargs: Any,
    ) -> Type:
        ...  # pragma: co cover

    @overload
    def get(  # type: ignore [misc]
        self,
        name: plugin_typing.PluginFullName,
        *args: Any,
        plugin_type: plugin_typing.PluginTypeArg = None,
        **kwargs: Any,
    ) -> Type:
        ...  # pragma: co cover

    def get(self, name: plugin_typing.PluginFullName, *args: Any, **kwargs: Any) -> Any:
        """Get a plugin by its full name (including category, i.e. of form
        ``'my_category.my_subcategory.my_plugin'``). Use ``*args`` and ``**kwargs`` to pass arguments to
        the plugin initializer. The returned object is an instance of the plugin class. If the plugin is not of the
        default plugin type, must provide ``plugin_type`` also.

        The method can be called with ``plugin_type`` and plugin initializer arguments, as follows:

        - As first positional argument after the plugin name:

        .. code-block:: python

            plugin_instance = get(
                "my_category.my_subcategory.my_plugin",  # Plugin full name.
                "method",  # Plugin type provided as a positional argument (first).
                0.4,  # First positional argument to plugin initializer.
                123,  # Second positional argument to plugin initializer...
                kwarg=2,  # Keyword argument(s) to plugin initializer from here on.
            )

        - As keyword argument:

        .. code-block:: python

            plugin_instance = get(
                "my_category.my_subcategory.my_plugin",  # Plugin full name.
                0.4,  # First positional argument to plugin initializer.
                123,  # Second positional argument to plugin initializer...
                plugin_type="method",  # Plugin type provided as a keyword argument.
                kwarg=2,  # Keyword argument(s) to plugin initializer from here on.
            )

        Args:
            name (~tempor.core.plugin.plugin_typing.PluginFullName):
                Plugin full name including all (sub)categories, of form ``'my_category.my_subcategory.my_plugin'``
            *args (Any):
                Arguments to pass to the plugin initializer.
            plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg, optional):
                Plugin type. If left as `None`, default plugin type is assumed. ``plugin_type`` must correctly \
                correspond to the category implied by plugin full name. Defaults to `None`.
            **kwargs (Any):
                Keyword arguments to pass to the plugin initializer.

        Returns:
            Any: The plugin instance initialised with ``*args`` and ``**kwargs`` as provided.
        """
        plugin_type, args, kwargs = self._handle_get_args_kwargs(args, kwargs)
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type, must_be_one_of=self.list_plugin_types())
        fqn = create_fqn(suffix=name, plugin_type=plugin_type)
        self._raise_plugin_does_not_exist_error(fqn)
        return self._plugin_registry[fqn](*args, **kwargs)

    def get_class(self, name: plugin_typing.PluginFullName, plugin_type: plugin_typing.PluginTypeArg = None) -> Type:
        """Get a plugin class (not instance) by its full name (including category, i.e. of form
        ``'my_category.my_subcategory.my_plugin'``). If the plugin is not of the default plugin type, must provide
        ``plugin_type``.

        Args:
            name (~tempor.core.plugin.plugin_typing.PluginFullName):
                Plugin full name including all (sub)categories, of form ``'my_category.my_subcategory.my_plugin'``
            plugin_type (~tempor.core.plugin.plugin_typing.PluginTypeArg, optional):
                Plugin type. If left as `None`, default plugin type is assumed. ``plugin_type`` must correctly \
                correspond to the category implied by plugin full name. Defaults to `None`.

        Returns:
            Type: Plugin class (not instance).
        """
        self._refresh()
        plugin_type = parse_plugin_type(plugin_type, must_be_one_of=self.list_plugin_types())
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
            init_file (str):
                The init file for the package directory containing the plugin modules (files).

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
            package_init_file (str):
                The init file for the package directory containing the plugin modules (files).

        Returns:
            List[str]: A list of plugin module names.
        """
        package_dir = os.path.dirname(package_init_file)
        paths = _glob_plugin_paths(package_dir=package_dir)
        return [_module_name_from_path(f) for f in paths]
