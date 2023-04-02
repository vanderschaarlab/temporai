import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.scaling import BaseScaler
from tempor.plugins.preprocessing.scaling.plugin_ts_standard_scaler import (
    TimeSeriesStandardScaler as plugin,
)
from tempor.utils.serialization import load, save

train_kwargs = {"random_state": 123}

TEST_ON_DATASETS = ["google_stocks_data_full", "sine_data_scaled_small"]
TEST_TRANSFORM_ON_DATASETS = ["sine_data_scaled_small"]


def from_api() -> BaseScaler:
    return plugin_loader.get("preprocessing.scaling.ts_standard_scaler", **train_kwargs)


def from_module() -> BaseScaler:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseScaler) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "ts_standard_scaler"
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseScaler, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
@pytest.mark.parametrize("data", TEST_TRANSFORM_ON_DATASETS)
def test_transform(test_plugin: BaseScaler, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    assert dataset.time_series is not None  # nosec B101
    assert (dataset.time_series.numpy() > 50).any()

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.transform(dataset)

    assert (output.time_series.numpy() < 50).all()
