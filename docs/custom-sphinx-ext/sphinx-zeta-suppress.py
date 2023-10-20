# type: ignore
# flake8: noqa
# pylint: skip-file

"""A Sphinx extension for more fine-grained control of Sphinx WARNING/ERROR log messages.

This is adapted directly from https://github.com/picnixz/sphinx-zeta-suppress with minimal modifications beyond
code formatting.

See resources:
- https://sphinx-zeta-suppress.readthedocs.io/
- https://github.com/picnixz/sphinx-zeta-suppress
- https://github.com/sphinx-doc/sphinx/issues/11325
"""

from __future__ import annotations

__all__ = ()

import abc
import contextlib
import importlib
import inspect
import logging
import pkgutil
import re
import warnings
from itertools import filterfalse, tee
from typing import TYPE_CHECKING, TypeVar

from sphinx.errors import ExtensionError
from sphinx.util.logging import NAMESPACE, SphinxLoggerAdapter, getLogger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator
    from types import ModuleType
    from typing import Any, Literal

    from sphinx.application import Sphinx
    from sphinx.config import Config
    from sphinx.extension import Extension
    from typing_extensions import TypeGuard

    T = TypeVar("T")

#: Logging level type.
Level = TypeVar("Level", int, str)

logger = getLogger(__name__)


def _notnone(value):
    # type: (Any) -> bool
    return value is not None


def _is_sphinx_logger_adapter(value):
    # type: (Any) -> TypeGuard[SphinxLoggerAdapter]
    return isinstance(value, SphinxLoggerAdapter)


def _is_pattern_like(obj):
    # type: (Any) -> TypeGuard[str | re.Pattern]
    return isinstance(obj, (str, re.Pattern))


def _partition(predicate, iterable):
    # type: (Callable[[T], bool], Iterable[T]) -> (Iterator[T], Iterator[T])
    """Partition an iterable into two iterators according to *predicate*.

    The result is `(no, yes)` of iterators such that *no* and *yes* iterate
    over the values in *iterable* for which *predicate* is falsey and truthy
    respectively.

    Typical usage::

        odd, even = partition(lambda x: x % 2 == 0, range(10))

        assert list(odd) == [1, 3, 5, 7, 8]
        assert list(even) == [0, 2, 4, 6, 8]
    """

    no, yes = tee(iterable)
    no, yes = filterfalse(predicate, no), filter(predicate, yes)
    return no, yes


def _normalize_level(level):
    # type: (Level) -> int | None
    """Convert a logging level name or integer into a known logging level."""
    if isinstance(level, int):
        return level

    try:
        # pylint: disable-next=W0212
        return logging._nameToLevel[level]
    except KeyError:
        return None
    except TypeError:
        raise TypeError(f"invalid logging level type for {level}")


def _parse_levels(levels):
    # type: (Level | list[Level] | tuple[Level, ...]) -> list[int]
    """Convert one or more logging levels into a list of logging levels."""
    if not isinstance(levels, (list, tuple)):
        if not isinstance(levels, (int, str)):
            raise TypeError("invalid logging level type")
        levels = [levels]
    return list(filter(_notnone, map(_normalize_level, levels)))


class SphinxSuppressFilter(logging.Filter, metaclass=abc.ABCMeta):
    def filter(self, record):
        # type: (logging.LogRecord) -> bool
        return not self.suppressed(record)

    @abc.abstractmethod
    def suppressed(self, record):
        # type: (logging.LogRecord) -> bool
        """Indicate whether *record* should be suppressed or not."""
        pass


class _All:
    """Container simulating the universe."""

    def __contains__(self, item):
        # type: (Any) -> Literal[True]
        return True


_ALL = _All()


class SphinxSuppressLogger(SphinxSuppressFilter):
    r"""A filter suppressing logging records issued by a Sphinx logger."""

    def __init__(self, name: str, levels=()):
        """
        Construct a :class:`SphinxSuppressLogger`.

        :param name: The (real) logger name to suppress.
        :type name: str
        :param levels: Optional logging levels to suppress.
        :type levels: bool | Level | list[Level] | tuple[Level, ...]
        """

        super().__init__(name)
        if isinstance(levels, bool):
            levels = _ALL if levels else []
        else:
            levels = _parse_levels(levels)

        #: List of logging levels to suppress.
        self.levels: list[int] = levels

    def suppressed(self, record):
        should_log = logging.Filter.filter(self, record)
        return not should_log or record.levelno in self.levels


class SphinxSuppressPatterns(SphinxSuppressFilter):
    r"""A filter suppressing matching messages."""

    def __init__(self, patterns=()):
        """
        Construct a :class:`SphinxSuppressPatterns`.

        :param patterns: Optional logging messages (regex) to suppress.
        :type patterns: list[str | re.Pattern]
        """

        super().__init__("")  # all loggers
        #: Set of patterns to search.
        self.patterns: set[re.Pattern] = set(map(re.compile, patterns))

    def suppressed(self, record):
        if not self.patterns:
            return False

        m = record.getMessage()
        return any(p.search(m) for p in self.patterns)


class SphinxSuppressRecord(SphinxSuppressLogger, SphinxSuppressPatterns):
    r"""A filter suppressing matching messages by logger's name pattern."""

    def __init__(self, name, levels=(), patterns=()):
        """
        Construct a :class:`SphinxSuppressRecord` filter.

        :param name: A logger's name to suppress.
        :type name: str
        :param levels: Optional logging levels to suppress.
        :type levels: bool | list[int]
        :param patterns: Optional logging messages (regex) to suppress.
        :type patterns: list[str | re.Pattern]
        """

        SphinxSuppressLogger.__init__(self, name, levels)
        SphinxSuppressPatterns.__init__(self, patterns)

    def suppressed(self, record):
        return SphinxSuppressLogger.suppressed(self, record) and SphinxSuppressPatterns.suppressed(self, record)


class _FiltersAdapter:
    def __init__(self, config):
        format_name = lambda name: f"{NAMESPACE}.{name}"

        filters_by_prefix = {}
        for name, levels in config.zeta_suppress_loggers.items():
            prefix = format_name(name)
            suppressor = SphinxSuppressLogger(prefix, levels)
            filters_by_prefix.setdefault(prefix, []).append(suppressor)

        suppress_records = config.zeta_suppress_records
        groups, patterns = _partition(_is_pattern_like, suppress_records)
        for group in groups:  # type: tuple[str, ...]
            prefix = format_name(group[0])
            suppressor = SphinxSuppressRecord(prefix, True, group[1:])
            filters_by_prefix.setdefault(prefix, []).append(suppressor)

        #: The filter to always add.
        self._global_filter = SphinxSuppressPatterns(patterns)
        #: The lists of filters to add, indexed by logger's prefix.
        #:
        #: The prefix always starts with :data:`sphinx.util.logging.NAMESPACE`,
        #: followed by a dot.
        self._filters_by_prefix = filters_by_prefix

    def get_module_names(self):
        # type: () -> Generator[str, None, None]
        """
        Yield the names of the modules that are expected to be altered.

        Note that the ``NAMESPACE + '.'`` prefix added by Sphinx is removed.
        """

        prefix_len = len(NAMESPACE) + 1
        for logger_name in self._filters_by_prefix:
            yield logger_name[prefix_len:]

    def get_filters(self, name):
        """Yield the filters to add for the given logger's name.

        :param name: The logger's name.
        :type name: str
        :return: The list of filters to add.
        :rtype: collections.abc.Generator[SphinxSuppressFilter, None, None]

        .. note:: The caller is responsible for adding the filters once.
        """

        for prefix, filters in self._filters_by_prefix.items():
            if name.startswith(prefix):
                yield from filters
        yield self._global_filter


_CACHE_ATTR_NAME = "_zeta_suppress_cache"


def _mark_module(app, module_name):
    # type: (Sphinx, str) -> None
    """Mark a module name as being altered."""
    getattr(app, _CACHE_ATTR_NAME).add(module_name)


def _skip_module(app, module_name):
    # type: (Sphinx, str) -> bool
    """Check whether a named module should be skipped or not."""
    return module_name in app.config.zeta_suppress_protect or module_name in getattr(app, _CACHE_ATTR_NAME)


def _update_module(config, module, filters):
    # type: (Config, ModuleType, _FiltersAdapter) -> None
    """Update the module's loggers using the corresponding filters."""
    adapters = inspect.getmembers(module, _is_sphinx_logger_adapter)
    for _, adapter in adapters:
        for f in filters.get_filters(adapter.logger.name):
            # Since a logger may be imported from a non-marked module,
            # we ensure that the filter is only added at most once.
            if f not in adapter.logger.filters:
                logger.debug("updating logger: %s", adapter.logger.name, type="sphinx-zeta-suppress", once=True)
                adapter.logger.addFilter(f)


@contextlib.contextmanager
def _suppress_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        yield


def _setup_filters(app, module, filters):
    """Alter the Sphinx loggers accessible in *module* and its submodules.

    :param app: The current Sphinx application.
    :type app: sphinx.application.Sphinx
    :param module: The module to alter.
    :type module: types.ModuleType
    :param filters: A filters configuration adapter.
    :type filters: _FiltersAdapter
    """

    if _skip_module(app, module.__name__):
        logger.debug("skipping module: %s", module.__name__, type="sphinx-zeta-suppress", once=True)
        return

    _mark_module(app, module.__name__)
    _update_module(app.config, module, filters)

    if not hasattr(module, "__path__"):
        return

    # scan the submodules
    mod_path, mod_prefix = module.__path__, module.__name__ + "."
    with _suppress_deprecation_warnings():
        for mod_info in pkgutil.walk_packages(mod_path, mod_prefix):
            if _skip_module(app, mod_info.name):
                logger.debug("skipping module: %s", mod_info.name, type="sphinx-zeta-suppress", once=True)
                continue

            try:
                submodule = importlib.import_module(mod_info.name)
            except ImportError as err:
                logger.warning("cannot import module: %s", mod_info.name, exc_info=err, type="sphinx-zeta-suppress")
                continue

            _mark_module(app, mod_info.name)
            _update_module(app, submodule, filters)


# event handlers


def install_suppress_handlers(app, config):
    # type: (Sphinx, Config) -> None
    """Event handler for :event:`config-inited`.

    This handler is called twice, namely as the first and the last handlers
    for :event:`config-inited` so that loggers emitting messages during the
    initialization or loggers declared after :event:`config-inited` is fired
    can be altered properly.
    """

    filters = _FiltersAdapter(config)

    # scan the loaded extensions and alter them
    for extension in app.extensions.values():  # type: Extension
        module = extension.module
        _setup_filters(app, module, filters)

    # scan modules and alter them directly
    with _suppress_deprecation_warnings():
        for mod_name in filters.get_module_names():  # type: str
            if not _skip_module(app, mod_name):
                try:
                    module = importlib.import_module(mod_name)
                except Exception as err:
                    msg = f"cannot import module {mod_name!r}"
                    raise ExtensionError(msg, err, __name__)

                _setup_filters(app, module, filters)


def _create_temporary_cache(app, config):
    # type: (Sphinx, Config) -> None
    """Create a temporary attribute to hold the altered modules."""
    if not hasattr(app, _CACHE_ATTR_NAME):
        setattr(app, _CACHE_ATTR_NAME, set())


def _delete_temporary_cache(app, config):
    # type: (Sphinx, Config) -> None
    """Delete the temporary attribute holding the altered modules."""
    if hasattr(app, _CACHE_ATTR_NAME):
        delattr(app, _CACHE_ATTR_NAME)


def setup(app):
    # type: (Sphinx) -> dict
    app.add_config_value("zeta_suppress_loggers", {}, True)
    app.add_config_value("zeta_suppress_protect", [], True)
    app.add_config_value("zeta_suppress_records", [], True)
    app.connect("config-inited", _create_temporary_cache, priority=-1)
    # @contract: no logger emits a message before 'config-inited' is fired
    app.connect("config-inited", install_suppress_handlers, priority=0)
    # @contract: no extension is loaded after config-inited is fired
    app.connect("config-inited", install_suppress_handlers, priority=1000)
    app.connect("config-inited", _delete_temporary_cache, priority=1001)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
