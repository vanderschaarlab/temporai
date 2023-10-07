import glob
import importlib
import importlib.abc
import importlib.util
import os
import os.path
import sys
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Tuple, Type, TypeVar, Union

from typing_extensions import ParamSpec

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
PluginType = Union[None, str]
PluginName = str
PluginFullName = str
PluginCategory = str
# Internal:
_PluginFqn = str
_PluginCategoryFqn = str

# Default plugin type:
DEFAULT_PLUGIN_TYPE = "method"


# Local helpers. ---


def get_default_plugin_type(plugin_type: PluginType) -> PluginType:
    """Get the default plugin type if ``plugin_type`` is ``None``.

    Args:
        plugin_type (PluginType): Plugin type.

    Returns:
        PluginType: Default plugin type if ``plugin_type`` is ``None``, otherwise ``plugin_type``.
    """
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
    name: ClassVar[PluginName] = PLUGIN_NAME_NOT_SET
    category: ClassVar[PluginCategory] = PLUGIN_CATEGORY_NOT_SET
    plugin_type: ClassVar[PluginType] = PLUGIN_TYPE_NOT_SET

    @classmethod
    def full_name(cls) -> str:
        """The full name of the plugin with its category: category.name"""
        return f"{cls.category}.{cls.name}"

    @classmethod
    def _fqn(cls) -> _PluginFqn:
        """The fully-qualified name of the plugin with its plugin type: [plugin_type].category.name"""
        return f"{create_fqn(cls.category, cls.plugin_type)}.{cls.name}"

    @classmethod
    def _category_fqn(cls) -> _PluginCategoryFqn:
        """The fully-qualified name of the plugin's category: [plugin_type].category"""
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
PLUGIN_REGISTRY: Dict[_PluginFqn, Type[Plugin]] = dict()


def register_plugin_category(category: PluginCategory, expected_class: Type, plugin_type: PluginType = None) -> None:
    plugin_type = get_default_plugin_type(plugin_type)
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
    def class_decorator(cls: Callable[P, T]) -> Callable[P, T]:
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

        _plugin_type = get_default_plugin_type(plugin_type)
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

    return class_decorator


# TODO: Add "list all" option, perhaps when "None" is passed in to plugin_type, in all the relevant listing methods.
# TODO: Add "list types".
# TODO: Add check plugin type exists before listing.
# TODO: Consider whether to enforce common base class across plugin_type/category.
class PluginLoader:
    def __init__(self) -> None:
        self._refresh()

    def _refresh(self):
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
        self._refresh()
        plugin_type = get_default_plugin_type(plugin_type)
        return self._plugin_name_by_category_nested[f"[{plugin_type}]"]

    def list_full_names(self, plugin_type: PluginType = None) -> List[PluginFullName]:
        self._refresh()
        plugin_fqns = list(self._plugin_registry.keys())
        plugin_type = get_default_plugin_type(plugin_type)
        plugin_fqns_filtered_by_type = filter_list_by_plugin_type(lst=plugin_fqns, plugin_type=plugin_type)
        return [remove_plugin_type_from_fqn(n) for n in plugin_fqns_filtered_by_type]

    def list_classes(self, plugin_type: PluginType = None) -> Dict:
        self._refresh()
        plugin_type = get_default_plugin_type(plugin_type)
        return self._plugin_class_by_category_nested[f"[{plugin_type}]"]

    def list_categories(self, plugin_type: PluginType = None) -> Dict[PluginFullName, Type[Plugin]]:
        self._refresh()
        plugin_type = get_default_plugin_type(plugin_type)
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
            plugin_type = get_default_plugin_type(plugin_type)
        return plugin_type, args, kwargs

    # TODO: Write type overloads.
    def get(self, name: PluginFullName, *args, **kwargs) -> Any:
        plugin_type, args, kwargs = self._handle_get_args_kwargs(args, kwargs)
        self._refresh()
        plugin_type = get_default_plugin_type(plugin_type)
        fqn = create_fqn(suffix=name, plugin_type=plugin_type)
        self._raise_plugin_does_not_exist_error(fqn)
        return self._plugin_registry[fqn](*args, **kwargs)

    def get_class(self, name: PluginFullName, plugin_type: PluginType = None) -> Type:
        self._refresh()
        plugin_type = get_default_plugin_type(plugin_type)
        fqn = create_fqn(suffix=name, plugin_type=plugin_type)
        self._raise_plugin_does_not_exist_error(fqn)
        return self._plugin_registry[fqn]


PLUGIN_FILENAME_PREFIX = "plugin_"


def _glob_plugin_paths(package_dir: str) -> List[str]:
    return [f for f in glob.glob(os.path.join(package_dir, f"{PLUGIN_FILENAME_PREFIX}*.py")) if os.path.isfile(f)]


def _module_name_from_path(path: str) -> str:
    path = os.path.normpath(path)
    split = path[path.rfind(f"{tempor.import_name}{os.sep}") :].split(os.sep)
    if split[-1][-3:] != ".py":  # pragma: no cover
        # Should be prevented by `_glob_plugin_paths`.
        raise ValueError(f"Path `{path}` is not a python file.")
    split[-1] = split[-1].replace(".py", "")
    return ".".join(split)


class importing:  # Functions as namespace, for clarity.
    @staticmethod
    def import_plugins(init_file: str) -> None:
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
        package_dir = os.path.dirname(package_init_file)
        paths = _glob_plugin_paths(package_dir=package_dir)
        return [_module_name_from_path(f) for f in paths]
