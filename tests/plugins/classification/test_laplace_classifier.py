import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.classification import BaseClassifier
from tempor.plugins.classification.plugin_laplace_classifier import (
    LaplaceODEClassifier as plugin,
)
from tempor.utils.dataloaders.google_stocks import GoogleStocksDataLoader
from tempor.utils.serialization import load, save

train_kwargs = {"random_state": 123, "n_iter": 50}


def from_api() -> BaseClassifier:
    return plugin_loader.get("classification.laplace_ode_classifier", **train_kwargs)


def from_module() -> BaseClassifier:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_laplace_ode_classifier_plugin_sanity(test_plugin: BaseClassifier) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "laplace_ode_classifier"
    assert test_plugin.fqn() == "classification.laplace_ode_classifier"
    assert len(test_plugin.hyperparameter_space()) == 7


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_laplace_ode_classifier_plugin_fit(test_plugin: BaseClassifier) -> None:
    dataset = GoogleStocksDataLoader().load()

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_laplace_ode_classifier_plugin_predict(test_plugin: BaseClassifier) -> None:
    dataset = GoogleStocksDataLoader().load()

    output = test_plugin.fit(dataset).predict(dataset)
    assert output.numpy().shape == (len(dataset.time_series), 1)


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin.sample_hyperparameters()
        plugin(**args)


def test_laplace_classifier_serde() -> None:
    test_plugin = from_api()

    data = GoogleStocksDataLoader().load()

    dump = save(test_plugin)
    reloaded1 = load(dump)

    reloaded1.fit(data)

    dump = save(reloaded1)
    reloaded2 = load(dump)

    reloaded2.predict(data)
