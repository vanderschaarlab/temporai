import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.regression import BaseRegressor
from tempor.plugins.regression.plugin_laplace_regressor import (
    LaplaceODERegressor as plugin,
)
from tempor.utils.dataloaders.google_stocks import GoogleStocksDataLoader
from tempor.utils.serialization import load, save

train_kwargs = {"random_state": 123, "n_iter": 50}


def from_api() -> BaseRegressor:
    return plugin_loader.get("regression.laplace_ode_regressor", **train_kwargs)


def from_module() -> BaseRegressor:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_laplace_ode_regressor_plugin_sanity(test_plugin: BaseRegressor) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "laplace_ode_regressor"
    assert len(test_plugin.hyperparameter_space()) == 7


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_laplace_ode_regressor_plugin_fit(test_plugin: BaseRegressor) -> None:
    dataset = GoogleStocksDataLoader().load()

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_laplace_ode_regressor_plugin_predict(test_plugin: BaseRegressor) -> None:
    dataset = GoogleStocksDataLoader().load()

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.predict(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 1)


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin.sample_hyperparameters()
        plugin(**args)
