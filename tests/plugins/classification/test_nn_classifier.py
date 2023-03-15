import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.classification import BaseClassifier
from tempor.plugins.classification.plugin_nn_classifier import (
    NeuralNetClassifier as plugin,
)
from tempor.utils.datasets.sine import SineDataloader


def from_api() -> BaseClassifier:
    return plugin_loader.get("classification.nn_classifier", random_state=123)


def from_module() -> BaseClassifier:
    return plugin(random_state=123)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nn_classifier_plugin_sanity(test_plugin: BaseClassifier) -> None:
    assert test_plugin is not None
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nn_classifier_plugin_fit(test_plugin: BaseClassifier) -> None:
    dataset = SineDataloader().load()

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nn_classifier_plugin_predict(test_plugin: BaseClassifier) -> None:
    dataset = SineDataloader().load()

    output = test_plugin.fit(dataset).predict(dataset)
    assert output.numpy().shape == (len(dataset.time_series), 1)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nn_classifier_plugin_predict_proba(test_plugin: BaseClassifier) -> None:
    dataset = SineDataloader().load()

    output = test_plugin.fit(dataset).predict_proba(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 2)


def test_hyperparam_sample():
    for repeat in range(100):
        args = plugin._cls.sample_hyperparameters()
        plugin(**args)
