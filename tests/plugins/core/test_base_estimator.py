# pylint: disable=redefined-outer-name, unused-argument

import dataclasses
import re
from typing import Any, Tuple
from unittest.mock import MagicMock, Mock

import pytest

import tempor.plugins.core


class TestBaseEstimator:
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
            ParamsDefinition = self.MyModelParams
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
            ParamsDefinition = self.MyModelParams

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

            ParamsDefinition = self.MyModelParamsWithNoDefault

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
            ParamsDefinition = self.MyModelParamsWithNoDefault

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
            ParamsDefinition = self.MyModelParams

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
            ParamsDefinition = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        with pytest.raises(ValueError, match=".*not.*float.*"):
            _ = MyModel(foo="string")

    def test_repr(self):
        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            ParamsDefinition = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel()
        repr_ = repr(my_model)

        assert re.search(r"^MyModel\(.*name='my_model'.*category='my_category'.*params=.?\{.*\}.*\)", repr_, re.S)

    def test_fit(self):
        from tempor.data.dataset import BaseDataset

        # ^ Import in this test only so others don't fail from problems in dataset.

        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            ParamsDefinition = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel()
        data = Mock(spec=BaseDataset)

        my_model.fit(data)

    def test_fit_fails_data_not_fit_ready(self):
        from tempor.data.dataset import BaseDataset

        # ^ Import in this test only so others don't fail from problems in dataset.

        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            ParamsDefinition = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel()
        data = Mock(spec=BaseDataset)
        data.fit_ready = False

        with pytest.raises(ValueError, match=".*not fit-ready.*"):
            my_model.fit(data)

    @pytest.mark.parametrize("override", [False, True])
    def test_sample_hyperparameters(self, override):
        hp = Mock()
        hp.name = "my_param"
        hp.sample = Mock(return_value="value")
        hps = [hp]

        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"
            ParamsDefinition = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return hps

            def _fit(self, data, *args, **kwargs):
                pass

        my_model = MyModel()
        if not override:
            sampled = my_model.sample_hyperparameters()
        else:
            override_hp = Mock()
            override_hp.name = "my_override_param"
            override_hp.sample = Mock(return_value="override_value")
            override_hps = [override_hp]
            sampled = my_model.sample_hyperparameters(override=override_hps)  # type: ignore

        if not override:
            assert sampled == {"my_param": "value"}
        else:
            assert sampled == {"my_override_param": "override_value"}
