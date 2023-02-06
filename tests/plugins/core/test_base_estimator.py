# pylint: disable=redefined-outer-name, unused-argument

import dataclasses
import re
from typing import Any, Tuple
from unittest.mock import MagicMock, Mock

import pytest

import tempor.plugins.core
from tempor.data import bundle


@pytest.fixture
def mock_data_bundle(monkeypatch):
    mock_data_bundle_ = Mock(spec=bundle.DataBundle, __name__="DataBundle")
    monkeypatch.setattr(bundle, "DataBundle", mock_data_bundle_)
    return mock_data_bundle_


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

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
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

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
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

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
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

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
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

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
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

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
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

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
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

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
                pass

        my_model = MyModel()
        repr_ = repr(my_model)

        assert re.search(r"^MyModel\(.*name='my_model'.*category='my_category'.*params=.?\{.*\}.*\)", repr_, re.S)

    def test_fit_validate_config(self, monkeypatch):
        # TODO: Deal with this later.

        import tempor.data.bundle.requirements as br
        import tempor.data.container.requirements as cr

        MockDataBundleValidator = Mock()
        MockTimeSeriesDataValidator = Mock()
        MockStaticDataValidator = Mock()
        MockEventDataValidator = Mock()
        monkeypatch.setattr(br, "DataBundleValidator", MockDataBundleValidator)
        monkeypatch.setattr(cr, "TimeSeriesDataValidator", MockTimeSeriesDataValidator)
        monkeypatch.setattr(cr, "StaticDataValidator", MockStaticDataValidator)
        monkeypatch.setattr(cr, "EventDataValidator", MockEventDataValidator)

        MockClass = Mock(spec=bundle.DataBundle, __name__="DataBundle")
        mock_from_data_containers = MagicMock(spec=bundle.DataBundle)
        mock_from_data_containers.get_time_series_containers = {"Xt": Mock()}
        mock_from_data_containers.get_static_containers = {"Xs": Mock(), "Ys": Mock()}
        mock_from_data_containers.get_event_containers = {"Xe": Mock(), "Ye": Mock(), "Ae": Mock()}
        MockClass.from_data_containers = Mock(return_value=mock_from_data_containers)
        monkeypatch.setattr(bundle, "DataBundle", MockClass)

        class MyModel(tempor.plugins.core.BaseEstimator):
            name = "my_model"
            category = "my_category"

            CONFIG = {
                "fit_config": {
                    "data_present": ["Xt", "Xs", "Xe", "Ys", "Ye", "Ae"],
                }
            }

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data: bundle.DataBundle, *args, **kwargs):
                pass

        my_model = MyModel()

        my_model.fit(data=mock_from_data_containers)

        MockDataBundleValidator.assert_called()
        MockTimeSeriesDataValidator.assert_called()
        assert MockTimeSeriesDataValidator.call_count == 1
        MockStaticDataValidator.assert_called()
        assert MockStaticDataValidator.call_count == 2
        MockEventDataValidator.assert_called()
        assert MockEventDataValidator.call_count == 3
