# pylint: disable=redefined-outer-name, unused-argument

from typing import Callable, Dict
from unittest.mock import Mock

import pytest

import tempor.core


def test_register_methods_success():
    mock_case_a_method = Mock()
    mock_case_b_method = Mock()

    class DummyOwnerClass:
        methods_for_stuff: Dict[str, Callable]

    class RegisterMethodDecoratorToTest(tempor.core.RegisterMethodDecorator):
        owner_class: type = DummyOwnerClass
        registration_dict_attribute_name: str = "methods_for_stuff"
        key_type: type = str
        method_category_name: str = "stuff"

    class DummyInheredClass(DummyOwnerClass):
        @RegisterMethodDecoratorToTest.register_method_for("case_a")  # pyright: ignore
        def _(self, dummy, args):
            mock_case_a_method(dummy=dummy, args=args)
            return "dummy return"

        @RegisterMethodDecoratorToTest.register_method_for("case_b")  # pyright: ignore
        def _(self, dummy, args):
            mock_case_b_method(dummy=dummy, args=args)
            return "dummy return"

    to_test = DummyInheredClass()

    assert "case_a" in to_test.methods_for_stuff
    assert "case_b" in to_test.methods_for_stuff

    to_test.methods_for_stuff["case_a"](self=to_test, dummy="dummy_a", args="args_a")
    to_test.methods_for_stuff["case_b"](self=to_test, dummy="dummy_b", args="args_b")

    mock_case_a_method.assert_called_once_with(dummy="dummy_a", args="args_a")
    mock_case_b_method.assert_called_once_with(dummy="dummy_b", args="args_b")


def test_fail_registered_multiple():
    class DummyOwnerClass:
        methods_for_stuff: Dict[str, Callable]

    class RegisterMethodDecoratorToTest(tempor.core.RegisterMethodDecorator):
        owner_class: type = DummyOwnerClass
        registration_dict_attribute_name: str = "methods_for_stuff"
        key_type: type = str
        method_category_name: str = "stuff"

    with pytest.raises(TypeError, match=".*registered multiple.*"):

        class DummyInheredClass(DummyOwnerClass):
            @RegisterMethodDecoratorToTest.register_method_for("case_a")  # pyright: ignore
            def _(self, dummy, args):
                return "dummy return"

            @RegisterMethodDecoratorToTest.register_method_for("case_a")  # pyright: ignore
            def _(self, dummy, args):
                return "dummy return"

        _ = DummyInheredClass()


def test_fail_wrong_owner_class():
    # Note, error raised in decorator won't be captured by pytest.raises, so using a workaround.

    import traceback

    class DummyOwnerClass:
        methods_for_stuff: Dict[str, Callable]

    class RegisterMethodDecoratorToTest(tempor.core.RegisterMethodDecorator):
        owner_class: type = DummyOwnerClass
        registration_dict_attribute_name: str = "methods_for_stuff"
        key_type: type = str
        method_category_name: str = "stuff"

    class WrongClass:
        pass

    try:

        class DummyInheredClass(WrongClass):
            @RegisterMethodDecoratorToTest.register_method_for("case_a")  # pyright: ignore
            def _(self, dummy, args):
                return "dummy return"

        _ = DummyInheredClass()

    except Exception:  # pylint: disable=broad-except
        ex_trace = traceback.format_exc()
        assert TypeError.__name__ in ex_trace
        assert "subclass of" in ex_trace


def test_inheritance_works():
    mock_parent_case_a_method = Mock()
    mock_parent_case_b_method = Mock()
    mock_child_case_a_method = Mock()
    mock_child_case_c_method = Mock()

    class DummyOwnerClass:
        methods_for_stuff: Dict[str, Callable]

    class RegisterMethodDecoratorToTest(tempor.core.RegisterMethodDecorator):
        owner_class: type = DummyOwnerClass
        registration_dict_attribute_name: str = "methods_for_stuff"
        key_type: type = str
        method_category_name: str = "stuff"

    class DummyInheredClass(DummyOwnerClass):
        @RegisterMethodDecoratorToTest.register_method_for("case_a")  # pyright: ignore
        def _(self, dummy, args):
            mock_parent_case_a_method(dummy=dummy, args=args)
            return "dummy return"

        @RegisterMethodDecoratorToTest.register_method_for("case_b")  # pyright: ignore
        def _(self, dummy, args):
            mock_parent_case_b_method(dummy=dummy, args=args)
            return "dummy return"

    class DummyInheredAgainClass(DummyInheredClass):
        @RegisterMethodDecoratorToTest.register_method_for("case_a")  # pyright: ignore
        def _(self, dummy, args):
            mock_child_case_a_method(dummy=dummy, args=args)
            return "dummy return"

        @RegisterMethodDecoratorToTest.register_method_for("case_c")  # pyright: ignore
        def _(self, dummy, args):
            mock_child_case_c_method(dummy=dummy, args=args)
            return "dummy return"

    to_test = DummyInheredAgainClass()

    to_test.methods_for_stuff["case_a"](self=to_test, dummy="dummy_a", args="args_a")
    to_test.methods_for_stuff["case_b"](self=to_test, dummy="dummy_b", args="args_b")
    to_test.methods_for_stuff["case_c"](self=to_test, dummy="dummy_c", args="args_c")

    mock_parent_case_a_method.assert_not_called()
    mock_parent_case_b_method.assert_called_once_with(dummy="dummy_b", args="args_b")
    mock_child_case_a_method.assert_called_once_with(dummy="dummy_a", args="args_a")
    mock_child_case_a_method.assert_called_once_with(dummy="dummy_a", args="args_a")
    mock_child_case_c_method.assert_called_once_with(dummy="dummy_c", args="args_c")


def test_missing_attributes():
    class DummyOwnerClass:
        methods_for_stuff: Dict[str, Callable]

    class RegisterMethodDecoratorToTest_MissingOwner(tempor.core.RegisterMethodDecorator):
        registration_dict_attribute_name: str = "methods_for_stuff"
        key_type: type = str
        method_category_name: str = "stuff"

    class RegisterMethodDecoratorToTest_MissingDict(tempor.core.RegisterMethodDecorator):
        owner_class: type = DummyOwnerClass
        key_type: type = str
        method_category_name: str = "stuff"

    class RegisterMethodDecoratorToTest_MissingKey(tempor.core.RegisterMethodDecorator):
        owner_class: type = DummyOwnerClass
        registration_dict_attribute_name: str = "methods_for_stuff"
        method_category_name: str = "stuff"

    class RegisterMethodDecoratorToTest_MissingName(tempor.core.RegisterMethodDecorator):
        owner_class: type = DummyOwnerClass
        registration_dict_attribute_name: str = "methods_for_stuff"
        key_type: type = str

    with pytest.raises(TypeError, match=".*owner_class.*"):

        class DummyInheredClass1(DummyOwnerClass):
            @RegisterMethodDecoratorToTest_MissingOwner.register_method_for("case_a")  # pyright: ignore
            def _(self, dummy, args):
                return "dummy return"

        _ = DummyInheredClass1()

    with pytest.raises(TypeError, match=".*registration_dict_attribute_name.*"):

        class DummyInheredClass2(DummyOwnerClass):
            @RegisterMethodDecoratorToTest_MissingDict.register_method_for("case_a")  # pyright: ignore
            def _(self, dummy, args):
                return "dummy return"

        _ = DummyInheredClass2()

    with pytest.raises(TypeError, match=".*key_type.*"):

        class DummyInheredClass3(DummyOwnerClass):
            @RegisterMethodDecoratorToTest_MissingKey.register_method_for("case_a")  # pyright: ignore
            def _(self, dummy, args):
                return "dummy return"

        _ = DummyInheredClass3()

    with pytest.raises(TypeError, match=".*method_category_name.*"):

        class DummyInheredClass4(DummyOwnerClass):
            @RegisterMethodDecoratorToTest_MissingName.register_method_for("case_a")  # pyright: ignore
            def _(self, dummy, args):
                return "dummy return"

        _ = DummyInheredClass4()
