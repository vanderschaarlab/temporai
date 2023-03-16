import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.classification import BaseClassifier
from tempor.plugins.classification.plugin_laplace_classifier import (
    LaplaceODEClassifier as plugin,
)
from tempor.utils.datasets.google_stocks import GoogleStocksDataloader

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
    dataset = GoogleStocksDataloader().load()

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_laplace_ode_classifier_plugin_predict(test_plugin: BaseClassifier) -> None:
    dataset = GoogleStocksDataloader().load()

    output = test_plugin.fit(dataset).predict(dataset)
    assert output.numpy().shape == (len(dataset.time_series), 1)


def test_hyperparam_sample():
    for repeat in range(100):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
