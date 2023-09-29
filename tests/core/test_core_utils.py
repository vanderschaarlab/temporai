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
        prefer="raise",
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
