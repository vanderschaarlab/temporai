import enum
from typing import Any, Dict, Tuple

from typing_extensions import get_args


def get_class_full_name(o: object):
    # See: https://stackoverflow.com/a/2020083
    class_ = o.__class__
    module = class_.__module__
    if module == "builtins":
        return class_.__qualname__  # avoid outputs like "builtins.str"
    return module + "." + class_.__qualname__


def get_enum_name(enum_: enum.Enum) -> str:
    return enum_.name.lower()


class RichReprStrPassthrough:
    def __init__(self, string: str) -> None:
        self.string = string

    def __repr__(self) -> str:
        return self.string


def is_iterable(o: object) -> bool:
    is_iterable_ = True
    try:
        iter(o)  # type: ignore[call-overload]
    except TypeError:
        is_iterable_ = False
    return is_iterable_


def get_version(version: str) -> Tuple[int, ...]:
    """Get the semantic ``version`` as a tuple of ``int`` s.

    Note:
        Assumes that the ``version`` string is specified as ``.``-separated ``ints``; will throw exceptions in
        case of more complex version semantics.

    Args:
        module (ModuleType): The module to get the version of.

    Returns:
        Tuple[int, ...]: Tuple of integers representing ``(major, minor, patch[, ...])``.
    """
    return tuple(int(v) for v in version.split("."))


def version_above_incl(version: Tuple[int, int], above_incl: Tuple[int, int]) -> bool:
    v_major, v_minor = version
    compare_major, compare_minor = above_incl
    return (v_major == compare_major and v_minor >= compare_minor) or v_major > compare_major


def version_below_excl(version: Tuple[int, int], below_excl: Tuple[int, int]) -> bool:
    v_major, v_minor = version
    compare_major, compare_minor = below_excl
    return (v_major == compare_major and v_minor < compare_minor) or v_major < compare_major


def ensure_literal_matches_dict_keys(
    literal: Any, d: Dict[str, Any], literal_name: str = "literal", dict_name: str = "dictionary"
):
    lits = set(get_args(literal))
    keys = set(d.keys())
    if lits != keys:
        raise TypeError(
            f"There was a mismatch between the literal '{literal_name}' and the the dictionary "
            f"'{dict_name}' keys: {list(lits.symmetric_difference(keys))}"
        )
