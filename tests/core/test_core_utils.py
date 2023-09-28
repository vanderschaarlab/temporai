import re

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


# Test get_from_args_or_kwargs ----------


class MyArgument:
    pass


def my_function(*args, **kwargs):
    value, args, kwargs = utils.get_from_args_or_kwargs(
        args, kwargs, argument_name="my_arg", argument_type=MyArgument, position_if_args=0
    )
    return value, args, kwargs


class TestGetFromArgsOrKwargs:
    def test_get_from_args(self):
        my_arg = MyArgument()

        value, args, kwargs = my_function(my_arg)
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple()
        assert kwargs == dict()

        value, args, kwargs = my_function(my_arg, "bar", "baz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple(["bar", "baz"])
        assert kwargs == dict()

        value, args, kwargs = my_function(my_arg, foobar="foobar", foobaz="foobaz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple()
        assert kwargs == dict(foobar="foobar", foobaz="foobaz")

        value, args, kwargs = my_function(my_arg, "bar", "baz", foobar="foobar", foobaz="foobaz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple(["bar", "baz"])
        assert kwargs == dict(foobar="foobar", foobaz="foobaz")

    def test_get_from_kwargs(self):
        my_arg = MyArgument()

        value, args, kwargs = my_function(my_arg=my_arg)
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple()
        assert kwargs == dict()

        value, args, kwargs = my_function("bar", "baz", my_arg=my_arg)
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple(["bar", "baz"])
        assert kwargs == dict()

        value, args, kwargs = my_function(my_arg=my_arg, foobar="foobar", foobaz="foobaz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple()
        assert kwargs == dict(foobar="foobar", foobaz="foobaz")

        value, args, kwargs = my_function("bar", "baz", my_arg=my_arg, foobar="foobar", foobaz="foobaz")
        assert value is my_arg
        assert value not in args and "my_arg" not in kwargs
        assert args == tuple(["bar", "baz"])
        assert kwargs == dict(foobar="foobar", foobaz="foobaz")

    def test_no_argument(self):
        value, args, kwargs = my_function()
        assert value is None
        assert args == tuple()
        assert kwargs == dict()

        value, args, kwargs = my_function(12345)
        assert value is None
        assert args == tuple([12345])
        assert kwargs == dict()

        value, args, kwargs = my_function(12345, "abc")
        assert value is None
        assert args == tuple([12345, "abc"])
        assert kwargs == dict()

        value, args, kwargs = my_function(foobar=999, foobaz="foobaz")
        assert args == tuple()
        assert kwargs == dict(foobar=999, foobaz="foobaz")

        value, args, kwargs = my_function("abc", -12345, foobar=999, foobaz="foobaz")
        assert value is None
        assert args == tuple(["abc", -12345])
        assert kwargs == dict(foobar=999, foobaz="foobaz")

    def test_exception_passed_as_both(self):
        my_arg = MyArgument()

        with pytest.raises(RuntimeError, match=".*as `kwargs`.*as `args`.*"):
            my_function(my_arg, "bar", my_arg=my_arg, foobaz=1234)

    def test_exception_kwargs_wrong_type(self):
        with pytest.raises(TypeError, match=".*`kwargs`.*type.*"):
            my_function("bar", "baz", my_arg="str", foobaz=1234)


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
