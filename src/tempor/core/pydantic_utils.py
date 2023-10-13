from typing import Any, Callable, Type, TypeVar, cast

import pydantic
from packaging.version import Version
from typing_extensions import ParamSpec

# Currently unused.
# def exclusive_args(
#     values: Dict,
#     arg1: str,
#     arg2: str,
#     arg1_friendly_name: Optional[str] = None,
#     arg2_friendly_name: Optional[str] = None,
# ) -> None:
#     arg1_value = values.get(arg1, None)
#     arg2_value = values.get(arg2, None)
#     arg1_name = arg1_friendly_name if arg1_friendly_name else f"`{arg1}`"
#     arg2_name = arg2_friendly_name if arg2_friendly_name else f"`{arg2}`"
#     if arg1_value is not None and arg2_value is not None:
#         raise ValueError(f"Must provide either {arg1_name} or {arg2_name} but not both")


def is_pydantic_dataclass(cls: Type) -> bool:
    if Version(pydantic.__version__) < Version("2.0.0"):  # pragma: no cover
        return hasattr(cls, "__dataclass__")
    else:
        return any([x for x in dir(cls) if "pydantic" in x])


PYDANTIC_DATACLASS_WORKAROUND_DICT = dict()


def make_pydantic_dataclass(builtin_dataclass: Type) -> Type:
    """Workaround for a `pydantic` edge case issue when calling ``pydantic.dataclass(<builtin_dataclass>)``
    more than once where ``builtin_dataclass`` has a default factory filed after a keyword parameter.

    E.g. the following would normally fail, this works around the issue.

    .. code-block:: python

        from typing import List
        import dataclasses
        import pydantic


        @dataclasses.dataclass
        class MyDataclass:
            a: str = "string"
            b: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])


        pydantic.dataclasses.dataclass(MyDataclass)  # OK.
        pydantic.dataclasses.dataclass(MyDataclass)  # TypeError.

    Args:
        builtin_dataclass (Type): Python builtin dataclass.

    Returns:
        Type: ``builtin_dataclass`` safely converted to pydantic dataclass.
    """
    name = f"{builtin_dataclass.__module__}.{builtin_dataclass.__name__}"
    if name not in PYDANTIC_DATACLASS_WORKAROUND_DICT:
        pydantic_dataclass: Any = pydantic.dataclasses.dataclass(builtin_dataclass)
        PYDANTIC_DATACLASS_WORKAROUND_DICT[name] = pydantic_dataclass
    else:
        pydantic_dataclass = PYDANTIC_DATACLASS_WORKAROUND_DICT[name]
    return pydantic_dataclass


P = ParamSpec("P")
T = TypeVar("T")


def validate_arguments(*args: Any, **kwargs: Any) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Uses the ``Callable[P, T]`` approach to type the pydantic ``validate_arguments`` decorator. Helps `mypy` to
    correctly understand typing of functions that are decorated by this.

    See:
    - https://stackoverflow.com/a/74080156
    - https://docs.python.org/3/library/typing.html#typing.ParamSpec

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: The updated ``pydantic.validate_arguments`` decorator.
    """
    return cast(
        Callable[P, T],  # type: ignore [valid-type, misc]
        pydantic.validate_arguments(*args, **kwargs),  # type: ignore [operator]
    )
