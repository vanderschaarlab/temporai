import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.regression import BaseRegressor
from tempor.plugins.regression.plugin_nn_regressor import NeuralNetRegressor as plugin
from tempor.utils.dataloaders.sine import SineDataLoader
from tempor.utils.serialization import load, save


def from_api() -> BaseRegressor:
    return plugin_loader.get("regression.nn_regressor", random_state=123)


def from_module() -> BaseRegressor:
    return plugin(random_state=123)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nn_regressor_plugin_sanity(test_plugin: BaseRegressor) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "nn_regressor"
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nn_regressor_plugin_fit(test_plugin: BaseRegressor) -> None:
    dataset = SineDataLoader().load()

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nn_regressor_plugin_predict(test_plugin: BaseRegressor) -> None:
    dataset = SineDataLoader().load()

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.predict(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 1)


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
