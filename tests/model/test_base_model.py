# pylint: disable=redefined-outer-name, unused-argument

import dataclasses
import re
from typing import Any, Tuple
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

import tempor.model
from tempor.data import bundle


@pytest.fixture
def mock_data_bundle(monkeypatch):
    mock_data_bundle_ = Mock(spec=bundle.DataBundle, __name__="DataBundle")
    monkeypatch.setattr(bundle, "DataBundle", mock_data_bundle_)
    return mock_data_bundle_


class TestTemporBaseModel:
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
        class MyModel(tempor.model.TemporBaseModel):
            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        my_model = MyModel()

        assert my_model.params == dict()
        assert my_model._fit_called is False  # pylint: disable=protected-access

    def test_init_success(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParams

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        my_model = MyModel()

        assert my_model.params == dict(foo=1.5, bar=(3, 7, 9), baz="something")

    def test_init_success_set_by_params(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParams

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        my_model = MyModel(params=dict(foo=2.2, bar=(8, 9, 10), baz="something_else"))

        assert my_model.params == dict(foo=2.2, bar=(8, 9, 10), baz="something_else")

    def test_init_success_set_by_kwargs(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParams

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        my_model = MyModel(foo=2.2, bar=(8, 9, 10), baz="something_else")

        assert my_model.params == dict(foo=2.2, bar=(8, 9, 10), baz="something_else")

    def test_init_fails_cannot_provide_params_in_both_ways(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParams

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        with pytest.raises(ValueError, match=".*not both.*"):
            _ = MyModel(params=dict(foo=2.2), baz="baz")

    def test_init_success_no_default_and_provided(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParamsWithNoDefault

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        my_model = MyModel(baz="provided")  # Does not fail.
        assert my_model.params == dict(foo=1.5, bar=(3, 7, 9), baz="provided")

    def test_init_fails_no_default_and_not_provided(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParamsWithNoDefault

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        with pytest.raises(ValueError) as excinfo:
            _ = MyModel()  # Fails.
        assert "missing" in str(excinfo.getrepr())

    def test_init_fails_omegaconf_incompatible_type(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParams

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        with pytest.raises(ValueError, match=".*not a supported primitive.*"):
            _ = MyModel(baz=Mock())

    def test_init_fails_wrong_type(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParams

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        with pytest.raises(ValueError, match=".*type 'str'.*"):
            _ = MyModel(foo="string")

    def test_repr(self):
        class MyModel(tempor.model.TemporBaseModel):
            PARAMS_DEFINITION = self.MyModelParams

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        my_model = MyModel()
        repr_ = repr(my_model)

        assert re.search(r"^MyModel(.*params=.?\{.*\})", repr_)

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            (
                [Mock(spec=pd.DataFrame, name="X")],
                dict(),
            ),
            (
                [
                    Mock(spec=pd.DataFrame, name="X"),
                    Mock(spec=pd.DataFrame, name="Y"),
                    Mock(spec=pd.DataFrame, name="A"),
                ],
                dict(),
            ),
            (
                [Mock(spec=pd.DataFrame, name="X")],
                {"Yt": Mock(spec=pd.DataFrame)},
            ),
            (
                [Mock(spec=pd.DataFrame, name="X")],
                {"Ys": Mock(spec=pd.DataFrame), "Yt": Mock(spec=pd.DataFrame), "Ae": Mock(spec=pd.DataFrame)},
            ),
            (
                [Mock(spec=pd.DataFrame, name="X"), Mock(spec=pd.DataFrame, name="Y")],
                {"As": Mock(spec=pd.DataFrame), "Ae": Mock(spec=pd.DataFrame)},
            ),
            (
                [],
                {"X": Mock(spec=pd.DataFrame), "Y": Mock(spec=pd.DataFrame)},
            ),
            (
                [],
                {"X": Mock(spec=pd.DataFrame), "Y": Mock(spec=pd.DataFrame), "A": Mock(spec=pd.DataFrame)},
            ),
            (
                [],
                {
                    "X": Mock(spec=pd.DataFrame),
                    "Ys": Mock(spec=pd.DataFrame),
                    "As": Mock(spec=pd.DataFrame),
                    "Ae": Mock(spec=pd.DataFrame),
                },
            ),
            (
                [],
                {
                    "X": Mock(spec=pd.DataFrame),
                    "A": Mock(spec=pd.DataFrame),
                    "Ys": Mock(spec=pd.DataFrame),
                    "Yt": Mock(spec=pd.DataFrame),
                    "Ye": Mock(spec=pd.DataFrame),
                },
            ),
            (
                [],
                {
                    "Xt": Mock(spec=pd.DataFrame),
                    "Ys": Mock(spec=pd.DataFrame),
                    "Ye": Mock(spec=pd.DataFrame),
                    "At": Mock(spec=pd.DataFrame),
                },
            ),
        ],
    )
    def test_fit_args_success(self, args, kwargs, mock_data_bundle: Mock):
        mock_validate_config = Mock()
        mock_fit = Mock()

        class MyModel(tempor.model.TemporBaseModel):
            def _fit(self, data: bundle.DataBundle, **kwargs):
                mock_fit()

            _validate_method_config = mock_validate_config

        my_model = MyModel()

        my_model.fit(*args, **kwargs)

        mock_data_bundle.from_data_containers.assert_called_once()
        mock_validate_config.assert_called_once()
        mock_fit.assert_called()
        assert my_model._fit_called is True  # pylint: disable=protected-access

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            (
                [Mock(spec=pd.DataFrame, name="X")],
                {"Xt": Mock(spec=pd.DataFrame)},
            ),
            (
                [
                    Mock(spec=pd.DataFrame, name="X"),
                    Mock(spec=pd.DataFrame, name="Y"),
                    Mock(spec=pd.DataFrame, name="A"),
                ],
                {"As": Mock(spec=pd.DataFrame), "At": Mock(spec=pd.DataFrame)},
            ),
            (
                [],
                {"X": Mock(spec=pd.DataFrame), "Xe": Mock(spec=pd.DataFrame)},
            ),
            (
                [],
                {"Y": Mock(spec=pd.DataFrame), "Ys": Mock(spec=pd.DataFrame), "Ye": Mock(spec=pd.DataFrame)},
            ),
            (
                [],
                {"X": Mock(spec=pd.DataFrame), "A": Mock(spec=pd.DataFrame), "Ae": Mock(spec=pd.DataFrame)},
            ),
        ],
    )
    def test_fit_args_fail_exclusive(self, args, kwargs, mock_data_bundle: Mock):
        class MyModel(tempor.model.TemporBaseModel):
            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        my_model = MyModel()

        with pytest.raises(ValueError, match=".*not both.*"):
            my_model.fit(*args, **kwargs)

    def test_fit_validate_config(self, monkeypatch):
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

        class MyModel(tempor.model.TemporBaseModel):
            CONFIG = {
                "fit_config": {
                    "data_present": ["Xt", "Xs", "Xe", "Ys", "Ye", "Ae"],
                }
            }

            def _fit(self, data: bundle.DataBundle, **kwargs):
                pass

        my_model = MyModel()

        my_model.fit(Xt=Mock(spec=pd.DataFrame))

        MockDataBundleValidator.assert_called()
        MockTimeSeriesDataValidator.assert_called()
        assert MockTimeSeriesDataValidator.call_count == 1
        MockStaticDataValidator.assert_called()
        assert MockStaticDataValidator.call_count == 2
        MockEventDataValidator.assert_called()
        assert MockEventDataValidator.call_count == 3
