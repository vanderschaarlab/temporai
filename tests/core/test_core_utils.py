# pylint: disable=unnecessary-pass

import re
from typing import Callable

import pytest
from typing_extensions import Literal

from tempor.core import utils


class TestEnsureLiteralMatchesDictKeys:
    @pytest.mark.parametrize(
        "lit, d",
        [
            (Literal["A", "B", "C"], {"A": None, "B": None, "C": None}),
            (Literal["A", "B", "C"], {"A": None, "B": None, "C": None}),
            (Literal["A"], {"A": None}),
        ],
    )
    def test_passes(self, lit, d):
        utils.ensure_literal_matches_dict_keys(lit, d)

    @pytest.mark.parametrize(
        "lit, d",
        [
            (Literal["A", "B", "C", "D"], {"A": None, "B": None, "C": None}),
            (Literal["A", "B", "X"], {"A": None, "B": None, "C": None}),
            (Literal["A", "B"], {"A": None, "B": None, "C": None}),
            (Literal["A"], {"B": None}),
            (Literal["A"], dict()),
        ],
    )
    def test_fails(self, lit, d):
        with pytest.raises(TypeError, match=".*MyLit.*MyDict.*"):
            utils.ensure_literal_matches_dict_keys(lit, d, literal_name="MyLit", dict_name="MyDict")

    @pytest.mark.parametrize(
        "lit, d, expect",
        [
            (Literal["A", "B", "C", "D"], {"A": None, "B": None, "C": None}, set(["D"])),
            (Literal["A", "C", "D"], {"A": None, "B": None, "C": None, "D": None}, set(["D"])),
            (Literal["A", "C", "D", "X"], {"A": None, "B": None, "C": None, "D": None}, set(["D", "X"])),
        ],
    )
    def test_fails_difference(self, lit, d, expect):
        with pytest.raises(TypeError, match=f".*{list(str(expect))}.*"):
            utils.ensure_literal_matches_dict_keys(lit, d)


# Test RichReprStrPassthrough ----------


def test_rich_repr_str_passthrough():
    rich_repr_passthrough = utils.RichReprStrPassthrough("foobar")
    assert rich_repr_passthrough.string == "foobar"
    assert repr(rich_repr_passthrough) == "foobar"


# Test RichReprStrPassthrough (end) ----------


# Test is_iterable ----------


def test_is_iterable():
    assert utils.is_iterable([1, 2, 3])
    assert utils.is_iterable("abc")
    assert utils.is_iterable(999) is False


# Test is_iterable (end) ----------


# Test get_from_args_or_kwargs ----------


class MyArgument:
    pass


def get_from_args_or_kwargs_preset_prefer_raise(*args, **kwargs):
    value, args, kwargs = utils.get_from_args_or_kwargs(
        args,
        kwargs,
        argument_name="my_arg",
        argument_type=MyArgument,
        position_if_args=0,
        prefer="exception",
    )
    return value, args, kwargs


def get_from_args_or_kwargs_preset_prefer_kwarg(*args, **kwargs):
    value, args, kwargs = utils.get_from_args_or_kwargs(
        args,
        kwargs,
        argument_name="my_arg",
        argument_type=MyArgument,
        position_if_args=0,
        prefer="kwarg",
    )
    return value, args, kwargs


def get_from_args_or_kwargs_preset_prefer_arg(*args, **kwargs):
    value, args, kwargs = utils.get_from_args_or_kwargs(
        args,
        kwargs,
        argument_name="my_arg",
        argument_type=MyArgument,
        position_if_args=0,
        prefer="arg",
    )
    return value, args, kwargs


class TestGetFromArgsOrKwargs:
    @pytest.mark.parametrize(
        "preset",
        [
            get_from_args_or_kwargs_preset_prefer_raise,
            get_from_args_or_kwargs_preset_prefer_arg,
            get_from_args_or_kwargs_preset_prefer_kwarg,
        ],
    )
    def test_get_from_args(self, preset: Callable):
        my_arg = MyArgument()

        value, args, kwargs = preset(my_arg)
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple()
        assert kwargs == dict()

        value, args, kwargs = preset(my_arg, "bar", "baz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple(["bar", "baz"])
        assert kwargs == dict()

        value, args, kwargs = preset(my_arg, foobar="foobar", foobaz="foobaz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple()
        assert kwargs == dict(foobar="foobar", foobaz="foobaz")

        value, args, kwargs = preset(my_arg, "bar", "baz", foobar="foobar", foobaz="foobaz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple(["bar", "baz"])
        assert kwargs == dict(foobar="foobar", foobaz="foobaz")

    @pytest.mark.parametrize(
        "preset",
        [
            get_from_args_or_kwargs_preset_prefer_raise,
            get_from_args_or_kwargs_preset_prefer_arg,
            get_from_args_or_kwargs_preset_prefer_kwarg,
        ],
    )
    def test_get_from_kwargs(self, preset: Callable):
        my_arg = MyArgument()

        value, args, kwargs = preset(my_arg=my_arg)
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple()
        assert kwargs == dict()

        value, args, kwargs = preset("bar", "baz", my_arg=my_arg)
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple(["bar", "baz"])
        assert kwargs == dict()

        value, args, kwargs = preset(my_arg=my_arg, foobar="foobar", foobaz="foobaz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple()
        assert kwargs == dict(foobar="foobar", foobaz="foobaz")

        value, args, kwargs = preset("bar", "baz", my_arg=my_arg, foobar="foobar", foobaz="foobaz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple(["bar", "baz"])
        assert kwargs == dict(foobar="foobar", foobaz="foobaz")

    @pytest.mark.parametrize(
        "preset",
        [
            get_from_args_or_kwargs_preset_prefer_raise,
            get_from_args_or_kwargs_preset_prefer_arg,
            get_from_args_or_kwargs_preset_prefer_kwarg,
        ],
    )
    def test_no_argument(self, preset: Callable):
        value, args, kwargs = preset()
        assert value is None
        assert args == tuple()
        assert kwargs == dict()

        value, args, kwargs = preset(12345)
        assert value is None
        assert args == tuple([12345])
        assert kwargs == dict()

        value, args, kwargs = preset(12345, "abc")
        assert value is None
        assert args == tuple([12345, "abc"])
        assert kwargs == dict()

        value, args, kwargs = preset(foobar=999, foobaz="foobaz")
        assert args == tuple()
        assert kwargs == dict(foobar=999, foobaz="foobaz")

        value, args, kwargs = preset("abc", -12345, foobar=999, foobaz="foobaz")
        assert value is None
        assert args == tuple(["abc", -12345])
        assert kwargs == dict(foobar=999, foobaz="foobaz")

    def test_exception_passed_as_both(self):
        my_arg = MyArgument()

        with pytest.raises(RuntimeError, match=".*as `kwargs`.*as `args`.*"):
            get_from_args_or_kwargs_preset_prefer_raise(my_arg, "bar", my_arg=my_arg, foobaz=1234)

    def test_prefer_kwarg_behavior(self):
        my_arg_passed_as_arg = MyArgument()
        my_arg_passed_as_kwarg = MyArgument()

        value, args, kwargs = get_from_args_or_kwargs_preset_prefer_kwarg(
            my_arg_passed_as_arg, "bar", 0.123, foobar="foobar", my_arg=my_arg_passed_as_kwarg, foobaz=1234
        )

        assert value == my_arg_passed_as_kwarg
        assert args == tuple([my_arg_passed_as_arg, "bar", 0.123])
        assert kwargs == dict(foobar="foobar", foobaz=1234)

    def test_prefer_arg_behavior(self):
        my_arg_passed_as_arg = MyArgument()
        my_arg_passed_as_kwarg = MyArgument()

        value, args, kwargs = get_from_args_or_kwargs_preset_prefer_arg(
            my_arg_passed_as_arg, "bar", 0.123, foobar="foobar", my_arg=my_arg_passed_as_kwarg, foobaz=1234
        )

        assert value == my_arg_passed_as_arg
        assert args == tuple(["bar", 0.123])
        assert kwargs == dict(foobar="foobar", my_arg=my_arg_passed_as_kwarg, foobaz=1234)

    @pytest.mark.parametrize(
        "preset",
        [
            get_from_args_or_kwargs_preset_prefer_raise,
            get_from_args_or_kwargs_preset_prefer_arg,
            get_from_args_or_kwargs_preset_prefer_kwarg,
        ],
    )
    def test_exception_kwargs_wrong_type(self, preset: Callable):
        with pytest.raises(TypeError, match=".*`kwargs`.*type.*"):
            preset("bar", "baz", my_arg="str", foobaz=1234)


# Test get_from_args_or_kwargs (end) -----


# Test get_class_full_name ----------


class MyClass:
    pass


my_class = MyClass()
my_str = str()


class TestGetClassFullName:
    def test_non_builtin(self):
        assert re.match(r".*test_core_utils\.MyClass", utils.get_class_full_name(my_class))

    def test_builtin(self):
        assert utils.get_class_full_name(my_str) == "str"


# Test get_class_full_name (end) ----------


# Test unique_in_order_of_appearance ----------


class TestUniqueInOrderOfAppearance:
    def test_strs(self):
        assert utils.unique_in_order_of_appearance(["a", "a", "b", "c", "c", "a", "b"]) == ["a", "b", "c"]

    def test_ints(self):
        assert utils.unique_in_order_of_appearance([1, 2, 2, 3, 3, 1, 1, 2]) == [1, 2, 3]

    def test_mixed(self):
        assert utils.unique_in_order_of_appearance([1, "a", 2, "a", 2, 3, 3, 1, 1, 2]) == [1, "a", 2, 3]


# Test unique_in_order_of_appearance (end) ----------


# Test is_method_defined_in_class ----------


class Parent:
    def __init__(self):
        pass

    def some_method(self):
        pass


class ChildInherits(Parent):
    pass


class ChildDoesNotInherit(Parent):
    def __init__(self) -> None:
        pass

    def some_method(self):
        pass


class TestIsMethodDefinedInClass:
    def test_yes_class(self):
        assert utils.is_method_defined_in_class(ChildDoesNotInherit, "__init__")
        assert utils.is_method_defined_in_class(ChildDoesNotInherit, "some_method")

    def test_no_class(self):
        assert utils.is_method_defined_in_class(ChildInherits, "__init__") is False
        assert utils.is_method_defined_in_class(ChildInherits, "some_method") is False

    def test_yes_instance(self):
        assert utils.is_method_defined_in_class(ChildDoesNotInherit(), "__init__")
        assert utils.is_method_defined_in_class(ChildDoesNotInherit(), "some_method")

    def test_no_instance(self):
        assert utils.is_method_defined_in_class(ChildInherits(), "__init__") is False
        assert utils.is_method_defined_in_class(ChildInherits(), "some_method") is False


# Test is_method_defined_in_class (end) ----------


# Test clean_multiline_docstr ----------


class ClassWithMultilineDocstr:
    """This is a docstring.
    It has multiple lines.
    """

    pass


class ClassWithSingleLineDocstr:
    """This is a docstring."""

    pass


class TestCleanMultilineDocstr:
    def test_multiline(self):
        assert (
            utils.clean_multiline_docstr(ClassWithMultilineDocstr.__doc__)  # pyright: ignore
            == "This is a docstring. It has multiple lines."
        )

    def test_single_line(self):
        assert (
            utils.clean_multiline_docstr(ClassWithSingleLineDocstr.__doc__) == "This is a docstring."  # pyright: ignore
        )


# Test clean_multiline_docstr (end) ----------


# Test make_description_from_doc ----------


class NoDocstr:
    pass


class DocstrClassOnly:
    """This is a docstring."""

    pass


class DocstrInitOnly:
    def __init__(self) -> None:
        """This is an __init__ docstring."""
        pass


class DocstrClassAndInit:
    """This is a docstring."""

    def __init__(self) -> None:
        """This is an __init__ docstring."""
        pass


class ParentWithInit:
    def __init__(self) -> None:
        """This is an __init__ docstring but it's on parent."""
        pass


class DocstrClassAndInitButInitOnlyFromParent:
    """This is a docstring."""

    pass


class TestMakeDescriptionFromDoc:
    def test_docstr_class_only(self):
        assert utils.make_description_from_doc(DocstrClassOnly()) == "This is a docstring."

    def test_init_only(self):
        assert utils.make_description_from_doc(DocstrInitOnly()) == "This is an __init__ docstring."

    def test_docstr_class_and_init(self):
        assert (
            utils.make_description_from_doc(DocstrClassAndInit())
            == "This is a docstring. This is an __init__ docstring."
        )

    def test_docstr_class_and_init_but_init_only_from_parent(self):
        assert utils.make_description_from_doc(DocstrClassAndInitButInitOnlyFromParent()) == "This is a docstring."

    def test_docstr_truncated(self):
        assert utils.make_description_from_doc(DocstrClassOnly(), max_len_keep=5) == "This ..."


# Test make_description_from_doc (end) ----------
