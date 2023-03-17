import enum


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
