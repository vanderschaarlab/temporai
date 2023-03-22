from typing import Any

import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.scaling import BaseScaler
from tempor.plugins.preprocessing.scaling.plugin_ts_minmax_scaler import (
    TimeSeriesMinMaxScaler as plugin,
)
from tempor.utils.dataloaders import GoogleStocksDataLoader, SineDataLoader


def from_api() -> BaseScaler:
    return plugin_loader.get("preprocessing.scaling.ts_minmax_scaler", random_state=123)


def from_module() -> BaseScaler:
    return plugin(random_state=123)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ts_minmax_scaler_plugin_sanity(test_plugin: BaseScaler) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "ts_minmax_scaler"
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
@pytest.mark.parametrize("dataloader", [GoogleStocksDataLoader(), SineDataLoader(ts_scale=5)])
def test_ts_minmax_scaler_plugin_fit(test_plugin: BaseScaler, dataloader: Any) -> None:
    dataset = dataloader.load()
    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ts_minmax_scaler_plugin_transform(test_plugin: BaseScaler) -> None:
    dataset = SineDataLoader(ts_scale=100).load()
    assert dataset.time_series is not None  # nosec B101
    assert (dataset.time_series.numpy() > 1.1).any()

    output = test_plugin.fit(dataset).transform(dataset)

    assert (output.time_series.numpy() < 1 + 1e-1).all()
    assert (output.time_series.numpy() >= 0).all()


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
