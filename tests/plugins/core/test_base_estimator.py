# pylint: disable=redefined-outer-name, unused-argument

import dataclasses
import re
from typing import Any, Tuple
from unittest.mock import MagicMock, Mock

import pytest

import tempor.plugins.core


class TestEstimator:
    @dataclasses.dataclass
    class MyModelParams:
        foo: float = 1.5
        bar: Tuple[int, ...] = (3, 7, 9)
        baz: Any = "something"

    @dataclasses.dataclass
    class MyModelParamsWithNoDefault:
        baz: Any  # No default
        foo: float = 1.5
        bar: Tuple[int, ...] = (3, 7, 9)

    def test_init_success_empty_params_definition_model(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel()

        assert my_model.params == dict()
        assert my_model._fitted is False  # pylint: disable=protected-access

    def test_init_success(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            PARAMS_DEFINITION = self.MyModelParams
            name = "my_model"
            category = "my_category"

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel()

        assert my_model.params == dict(foo=1.5, bar=(3, 7, 9), baz="something")

    def test_init_success_params_provided(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            PARAMS_DEFINITION = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel(foo=2.2, bar=(8, 9, 10), baz="something_else")

        assert my_model.params == dict(foo=2.2, bar=(8, 9, 10), baz="something_else")

    def test_init_success_no_default_and_provided(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"

            PARAMS_DEFINITION = self.MyModelParamsWithNoDefault

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel(baz="provided")  # Does not fail.
        assert my_model.params == dict(foo=1.5, bar=(3, 7, 9), baz="provided")

    def test_init_fails_no_default_and_not_provided(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            PARAMS_DEFINITION = self.MyModelParamsWithNoDefault

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        with pytest.raises(ValueError) as excinfo:
            _ = MyModel()  # Fails.
        assert "missing" in str(excinfo.getrepr())

    def test_init_fails_omegaconf_incompatible_type(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            PARAMS_DEFINITION = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        with pytest.raises(ValueError, match=".*not a supported primitive.*"):
            _ = MyModel(baz=Mock())

    def test_init_fails_wrong_type(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            PARAMS_DEFINITION = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        with pytest.raises(ValueError, match=".*type 'str'.*"):
            _ = MyModel(foo="string")

    def test_repr(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            PARAMS_DEFINITION = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel()
        repr_ = repr(my_model)

        assert re.search(r"^MyModel\(.*name='my_model'.*category='my_category'.*params=.?\{.*\}.*\)", repr_, re.S)
