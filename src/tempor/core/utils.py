from typing import Any, Dict, Tuple, Type

from typing_extensions import Literal, get_args


def get_class_full_name(o: object):
    # See: https://stackoverflow.com/a/2020083
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


PreferArgOrKwarg = Literal["arg", "kwarg", "raise"]


def get_from_args_or_kwargs(
    args: Tuple,
    kwargs: Dict,
    argument_name: str,
    argument_type: Type,
    position_if_args: int,
    prefer: PreferArgOrKwarg = "raise",
) -> Tuple[Any, Tuple, Dict]:
    """Will attempt to get the function argument as defined by ``argument_name``, ``argument_type``, and
    ``position_if_args`` from ``args`` and ``kwargs``. Will return `None` if no such argument found.
    Will raise an exception if ``args`` and ``kwargs`` have a problem with the definition given.

    Algorithm:
        1. Check if an ``arg`` of type ``argument_type`` is found at index ``position_if_args`` in ``args``.
        2. Check if a ``kwarg`` by key ``argument_name`` is found in ``kwargs``.
        3. If both 1 and 2 are found raise `RuntimeError` if ``prefer`` is set to ``"raise"``. Otherwise take the \
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
            to ``"raise"``. Defaults to ``"raise"``.

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
        if prefer == "raise":
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
