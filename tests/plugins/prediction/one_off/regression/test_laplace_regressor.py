import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.prediction.one_off.regression import BaseOneOffRegressor
from tempor.plugins.prediction.one_off.regression.plugin_laplace_regressor import (
    LaplaceODERegressor as plugin,
)
from tempor.utils.serialization import load, save

train_kwargs = {"random_state": 123, "n_iter": 5}

TEST_ON_DATASETS = ["google_stocks_data_small"]


def from_api() -> BaseOneOffRegressor:
    return plugin_loader.get("prediction.one_off.regression.laplace_ode_regressor", **train_kwargs)


def from_module() -> BaseOneOffRegressor:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseOneOffRegressor) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "laplace_ode_regressor"
    assert len(test_plugin.hyperparameter_space()) == 7


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseOneOffRegressor, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_predict(test_plugin: BaseOneOffRegressor, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.predict(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 1)
