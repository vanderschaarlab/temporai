import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.regression import BaseRegressor
from tempor.plugins.regression.plugin_ode_regressor import ODERegressor as plugin
from tempor.utils.serialization import load, save

train_kwargs = {"random_state": 123, "n_iter": 5}

TEST_ON_DATASETS = ["sine_data_small"]


def from_api() -> BaseRegressor:
    return plugin_loader.get("regression.ode_regressor", **train_kwargs)


def from_module() -> BaseRegressor:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseRegressor) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "ode_regressor"
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseRegressor, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_predict(test_plugin: BaseRegressor, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.predict(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 1)
