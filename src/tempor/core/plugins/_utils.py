"""Module with utilities that support plugin functionality."""

from typing import Any, Dict, List

# Helpers for organizing the plugin registry.


def add_by_list_of_keys(dictionary: Dict, key_path: List[Any], value: Any) -> Dict:
    """Add `value` to (nested) `dictionary` as specified by `key_path` (a list of nested keys)."""
    key = key_path[0]
    dictionary[key] = (
        value
        if len(key_path) == 1
        else add_by_list_of_keys(
            dictionary[key] if key in dictionary else dict(),
            key_path[1:],
            value,
        )
    )
    return dictionary


def get_by_list_of_keys(dictionary: Dict, key_path: List[Any]) -> Dict:
    """Get value within (nested) `dictionary` as specified by `key_path` (a list of nested keys)."""
    if len(key_path) == 1:
        return dictionary[key_path[0]]
    else:
        return get_by_list_of_keys(dictionary[key_path[0]], key_path[1:])


def add_by_dot_path(dictionary: Dict, key_path: str, value: Any) -> Dict:
    """Add `value` to (nested) `dictionary` as specified by `key_path` (a dot separated string of nested keys)."""
    return add_by_list_of_keys(dictionary, key_path.split("."), value)


def get_by_dot_path(dictionary: Dict, key_path: str) -> Any:
    """Get value within (nested) `dictionary` as specified by `key_path` (a dot separated string of nested keys)."""
    return get_by_list_of_keys(dictionary, key_path.split("."))


def append_by_dot_path(dictionary: Dict, key_path: str, value: Any) -> Dict:
    """Append `value` to a list stored in a (nested) `dictionary` as specified by `key_path` (a dot separated string of
    nested keys).

    Expects item in `dictionary` at `key_path` to be a list, so that `value` can be appended.

    If no `key_path` key found, create a list at `key_path` and put `value` into it.
    """
    try:
        get_by_dot_path(dictionary, key_path).append(value)
    except KeyError:
        add_by_dot_path(dictionary, key_path, [value])
    return dictionary
