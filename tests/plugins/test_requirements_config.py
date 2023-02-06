# pylint: disable=redefined-outer-name, unused-argument

import re
from typing import Any
from unittest.mock import Mock

import pytest

import tempor.core.requirements as r
import tempor.data._settings as dat_settings
import tempor.data._types as dat_types
from tempor.plugins import _requirements_config as rc
from tempor.plugins.core import _types as types


@pytest.fixture
def mock_requirements(monkeypatch):
    monkeypatch.setattr(r, "Requirement", Mock())


dummy_data_present_arg: Any = ["Xt"]


class TestRequirementsConfig:
    def test_init_fit_config(self):
        config = rc.RequirementsConfig(
            fit_config=Mock(spec=rc.FitConfig),
        )

        assert config.fit_config is not None
        assert config.transform_config is None
        assert config.predict_config is None
        assert config.predict_counterfactual_config is None

    def test_init_all_method_configs(self):
        config = rc.RequirementsConfig(
            fit_config=Mock(spec=rc.FitConfig),
            transform_config=Mock(spec=rc.TransformConfig),
            predict_config=Mock(spec=rc.PredictConfig),
            predict_counterfactual_config=Mock(spec=rc.PredictCounterfactualConfig),
        )

        assert config.fit_config is not None
        assert config.transform_config is not None
        assert config.predict_config is not None
        assert config.predict_counterfactual_config is not None

    def test_get_requirements(self):
        config = rc.RequirementsConfig(
            fit_config=Mock(spec=rc.FitConfig),
        )

        reqs = config.get_requirements()
        assert reqs.data_bundle_requirements == []
        assert reqs.data_container_requirements == []

    def test_str(self):
        config = rc.RequirementsConfig(
            fit_config=Mock(spec=rc.FitConfig),
            transform_config=Mock(spec=rc.TransformConfig),
            predict_config=Mock(spec=rc.PredictConfig),
            predict_counterfactual_config=Mock(spec=rc.PredictCounterfactualConfig),
        )

        str_ = str(config)

        assert re.search(r"^RequirementsConfig(.*)", str_)


@pytest.fixture()
def MethodConfigUnderTest():
    class MethodConfig(rc._MethodConfig):  # pylint: disable=protected-access
        @property
        def method_type(self) -> types.EstimatorMethods:
            return types.EstimatorMethods.FIT

    return MethodConfig


class TestMethodConfig:
    def test_init_success(self, MethodConfigUnderTest):
        method_config: rc._MethodConfig = MethodConfigUnderTest(
            data_present=dummy_data_present_arg, Xt_config=Mock(spec=rc.TimeSeriesDataContainerConfig)
        )

        assert method_config.method_type == types.EstimatorMethods.FIT  # noqa: E721
        assert method_config.Xt_config is not None
        assert method_config.Xs_config is None
        assert method_config.Xe_config is None
        assert method_config.Yt_config is None
        assert method_config.Ys_config is None
        assert method_config.Ye_config is None
        assert method_config.At_config is None
        assert method_config.As_config is None
        assert method_config.Ae_config is None

    def test_defaults_get_set(self, MethodConfigUnderTest, monkeypatch):
        MockTimeSeriesDataContainerConfig = Mock(spec=rc.TimeSeriesDataContainerConfig)
        MockStaticDataContainerConfig = Mock(spec=rc.StaticDataContainerConfig)
        MockEventDataContainerConfig = Mock(spec=rc.EventDataContainerConfig)
        monkeypatch.setattr(
            rc,
            "METHOD_CONFIG_DEFAULTS_DISPATCH",
            {
                MockTimeSeriesDataContainerConfig: ("Xt", "Yt", "At"),
                MockStaticDataContainerConfig: ("Xs", "Ys", "As"),
                MockEventDataContainerConfig: ("Xe", "Ye", "Ae"),
            },
        )

        method_config: rc._MethodConfig = MethodConfigUnderTest(
            data_present=["Xt", "Xe", "Xs", "Yt", "Ye", "Ys", "At", "Ae", "As"],
            Xt_config=Mock(spec=rc.TimeSeriesDataContainerConfig),
        )

        assert MockTimeSeriesDataContainerConfig.call_count == 2
        assert MockStaticDataContainerConfig.call_count == 3
        assert MockEventDataContainerConfig.call_count == 3

        assert method_config.Xt_config is not None
        assert method_config.Xs_config is not None
        assert method_config.Xe_config is not None
        assert method_config.Yt_config is not None
        assert method_config.Ys_config is not None
        assert method_config.Ye_config is not None
        assert method_config.At_config is not None
        assert method_config.As_config is not None
        assert method_config.Ae_config is not None

    def test_get_reqs(self, MethodConfigUnderTest, mock_requirements):
        method_config: rc._MethodConfig = MethodConfigUnderTest(
            data_present=dummy_data_present_arg, Xt_config=Mock(spec=rc.TimeSeriesDataContainerConfig)
        )

        reqs: rc.RequirementsSet = method_config.get_requirements()

        assert not reqs.data_container_requirements


class TestFitConfig:
    def test_init_success(self):
        config = rc.FitConfig()

        assert config.method_type == types.EstimatorMethods.FIT  # noqa: E721


class TestTransformConfig:
    def test_init_success(self):
        config = rc.TransformConfig(data_present=dummy_data_present_arg)

        assert config.data_present == dummy_data_present_arg
        assert config.method_type == types.EstimatorMethods.TRANSFORM  # noqa: E721


class TestPredictConfig:
    def test_init_success(self):
        config = rc.PredictConfig(data_present=dummy_data_present_arg)

        assert config.data_present == dummy_data_present_arg
        assert config.method_type == types.EstimatorMethods.PREDICT  # noqa: E721


class TestPredictCounterfactualConfig:
    def test_init_success(self):
        config = rc.PredictCounterfactualConfig(data_present=dummy_data_present_arg)

        assert config.data_present == dummy_data_present_arg
        assert config.method_type == types.EstimatorMethods.PREDICT_COUNTERFACTUAL  # noqa: E721


MockDataCategory = Mock()


@pytest.fixture
def DataContainerConfigUnderTest():
    class DataContainerConfig(rc._DataContainerConfig):  # pylint: disable=protected-access
        @property
        def data_category(self) -> dat_types.DataCategory:
            return MockDataCategory

    return DataContainerConfig


class TestDataContainerConfig:
    def test_init(self, DataContainerConfigUnderTest):
        container_config: rc._DataContainerConfig = DataContainerConfigUnderTest()

        assert container_config.data_category == MockDataCategory

        # Check default values:
        assert set(container_config.value_dtypes) == dat_settings.DATA_SETTINGS.value_dtypes
        assert container_config.allow_missing is True

    def test_get_reqs(self, DataContainerConfigUnderTest, mock_requirements):
        container_config: rc._DataContainerConfig = DataContainerConfigUnderTest()
        reqs: rc.RequirementsSet = container_config.get_requirements()

        assert reqs.data_container_requirements
        assert not reqs.data_bundle_requirements


class TestTimeSeriesDataContainerConfig:
    def test_init(self):
        config = rc.TimeSeriesDataContainerConfig()

        assert config.data_category == dat_types.DataCategory.TIME_SERIES


class TestStaticDataContainerConfig:
    def test_init(self):
        config = rc.StaticDataContainerConfig()

        assert config.data_category == dat_types.DataCategory.STATIC


class TestEventDataContainerConfig:
    def test_init(self):
        config = rc.EventDataContainerConfig()

        assert config.data_category == dat_types.DataCategory.EVENT
