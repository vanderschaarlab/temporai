"""Utility functions for TemporAI core."""

from typing import Any, Dict, Iterable, List, Tuple, Type

from typing_extensions import Literal, get_args


def get_class_full_name(o: object) -> str:
    """Get the full name of a class.

    See: https://stackoverflow.com/a/2020083.

    Args:
        o (object): The object to get the class full name of.

    Returns:
        str: The full name of the class.
    """
    class_ = o.__class__
    module = class_.__module__
    if module == "builtins":
        return class_.__qualname__  # avoid outputs like "builtins.str"
    return module + "." + class_.__qualname__


# Currently unused
# def get_enum_name(enum_: enum.Enum) -> str:
#     return enum_.name.lower()


class RichReprStrPassthrough:
    def __init__(self, string: str) -> None:
        """A pass-through class for `rich` ``__repr__`` strings. Yields the ``string`` in its ``__repr__``.

        Args:
            string (str): The string to pass through.
        """
        self.string = string

    def __repr__(self) -> str:
        """The ``__repr__`` method, will simply return the ``string`` provided at initialization.

        Returns:
            str: String to return.
        """
        return self.string


def is_iterable(o: object) -> bool:
    """Check if an object is an iterable.

    Args:
        o (object): The object to check.

    Returns:
        bool: Whether the object is an iterable.
    """
    is_iterable_ = True
    try:
        iter(o)  # type: ignore[call-overload]
    except TypeError:
        is_iterable_ = False
    return is_iterable_


def ensure_literal_matches_dict_keys(
    literal: Any, d: Dict[str, Any], literal_name: str = "literal", dict_name: str = "dictionary"
) -> None:
    """Check that the args of a literal match the keys of a dictionary.

    Args:
        literal (Any): A literal.
        d (Dict[str, Any]): A dictionary.
        literal_name (str, optional): The name of the literal, for exception description. Defaults to ``"literal"``.
        dict_name (str, optional): The name of the dictionary, for exception description. Defaults to ``"dictionary"``.

    Raises:
        TypeError: Raised if the args of the literal do not match the keys of the dictionary.
    """
    lits = set(get_args(literal))
    keys = set(d.keys())
    if lits != keys:
        raise TypeError(
            f"There was a mismatch between the literal '{literal_name}' and the the dictionary "
            f"'{dict_name}' keys: {list(lits.symmetric_difference(keys))}"
        )


PreferArgOrKwarg = Literal["arg", "kwarg", "exception"]
"""Literal type for ``prefer`` argument in ``get_from_args_or_kwargs``.
One of ``"arg"``, ``"kwarg"``, or ``"exception"``.
"""


def get_from_args_or_kwargs(
    args: Tuple,
    kwargs: Dict,
    argument_name: str,
    argument_type: Type,
    position_if_args: int,
    prefer: PreferArgOrKwarg = "exception",
) -> Tuple[Any, Tuple, Dict]:
    """Will attempt to get the function argument as defined by ``argument_name``, ``argument_type``, and
    ``position_if_args`` from ``args`` and ``kwargs``. Will return `None` if no such argument found.

    Algorithm:
        1. Check if an ``arg`` of type ``argument_type`` is found at index ``position_if_args`` in ``args``.
        2. Check if a ``kwarg`` by key ``argument_name`` is found in ``kwargs``.
        3. If both 1 and 2 are found raise `RuntimeError` if ``prefer`` is set to ``"exception"``. Otherwise take the \
            ``arg`` or ``kwarg`` item as specified by ``prefer`` accordingly. The other item will be left as it was \
            originally provided in ``args``/``kwargs``.
        4. If ``kwarg`` from 2 is not of type ``argument_type`` raise `TypeError`.
        5. Return 1 or 2 if argument is found, else return `None`. Also return ``args`` and ``kwargs``\
            with the argument "popped".

    Args:
        args (Tuple):
            ``args`` to check.
        kwargs (Dict):
            ``kwargs`` to check.
        argument_name (str):
            The name of the argument to look for.
        argument_type (Type):
            The type of the argument to confirm.
        position_if_args (int):
            The index in ``args`` at which the argument should be found, if it is provided by ``args``.
        prefer (PreferArgOrKwarg, optional):
            Whether to prefer the ``arg`` or the ``kwarg`` if both are found, or to raise an exception if this is set \
            to ``"exception"``. Defaults to ``"exception"``.

    Raises:
        RuntimeError: Error in case the argument appears to have been provided by both ``args`` and ``kwargs``.
        TypeError: Error in case the ``kwarg`` provided by key ``argument_name`` is not of the expected type.

    Returns:
        Tuple[Any, Tuple, Dict]:
            ``(found_argument_or_None, args, kwargs)``.
            If argument found, it will be removed from the ``args``/``kwargs`` returned.
    """
    from_args = None
    if len(args) >= (position_if_args + 1):
        arg_at_position = args[position_if_args]
        if isinstance(arg_at_position, argument_type):
            from_args = arg_at_position
            args = tuple([x for i, x in enumerate(args) if i != position_if_args])
    from_kwargs = kwargs.pop(argument_name, None)
    if from_args is not None and from_kwargs is not None:
        if prefer == "exception":
            raise RuntimeError(
                f"Argument `{argument_name}` appears to have been passed as `kwargs` (by key '{argument_name}') "
                f"and as `args` (at position {position_if_args}), but it should be passed in only one of these ways"
            )
        elif prefer == "kwarg":
            args_list = list(args)
            args_list.insert(position_if_args, from_args)
            args = tuple(args_list)
            from_args = None
        else:
            kwargs[argument_name] = from_kwargs
            from_kwargs = None
    if from_kwargs is not None and not isinstance(from_kwargs, argument_type):
        raise TypeError(
            f"Argument `{argument_name}` was passed as `kwargs` (by key '{argument_name}') but was not of "
            f"expected type `{argument_type}`"
        )
    return from_args if from_args is not None else from_kwargs, args, kwargs


def unique_in_order_of_appearance(iterable: Iterable) -> List:
    """Return unique elements from ``iterable`` in order of their appearance.

    Note:
        All items in ``iterable`` must be hashable.

    Args:
        iterable (Iterable): The iterable to get unique elements from.

    Returns:
        List: List of unique elements in order of their appearance.
    """
    return list(dict.fromkeys(iterable))


def is_method_defined_in_class(cls_or_obj: Any, method_name: str) -> bool:
    """Check if method named ``method_name`` method is defined in the given class (or object's class) or inherited.

    Args:
        cls_or_obj (Any): The class or object to check.
        method_name (str): The name of the method to check.

    Returns:
        bool: True if method is defined in ``cls``, `False` if inherited.
    """
    init_qualname = getattr(cls_or_obj, method_name).__qualname__
    class_name = cls_or_obj.__name__ if isinstance(cls_or_obj, type) else cls_or_obj.__class__.__name__
    return init_qualname.startswith(class_name + ".")


def clean_multiline_docstr(docstr: str) -> str:
    """Clean a multi-line docstring by getting rid of newlines and cleaning up whitespace.

    Args:
        docstr (str): The docstring to clean.

    Returns:
        str: The cleaned docstring.
    """
    return " ".join([line.strip() for line in docstr.split("\n")]).strip()


def make_description_from_doc(obj: Any, max_len_keep: int = 100) -> str:
    """Make a description from the docstring of an object. Take the class docstring and the ``__init__`` docstring
    (if there was one defined on the class). If the combined length of these is greater than ``max_len_keep``, then
    truncate with ``...``.

    Args:
        obj (Any): The object to get the description of.
        max_len_keep (int, optional): Maximum description length before truncating. Defaults to ``100``.

    Returns:
        str: Description of the object.
    """
    class_doc: str = obj.__doc__ if obj.__doc__ is not None else ""
    init_doc: str = ""
    if is_method_defined_in_class(obj, "__init__"):
        init_doc = obj.__init__.__doc__ if obj.__init__.__doc__ is not None else ""
    class_and_init_docs_combo = f"{class_doc} {init_doc}".strip()

    if len(class_and_init_docs_combo) > max_len_keep:
        return class_and_init_docs_combo[:max_len_keep] + "..."
    else:
        return class_and_init_docs_combo
