import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.classification import BaseClassifier
from tempor.plugins.classification.plugin_cde_classifier import CDEClassifier as plugin
from tempor.utils.datasets.sine import SineDataloader

train_kwargs = {"random_state": 123, "n_iter": 50}


def from_api() -> BaseClassifier:
    return plugin_loader.get("classification.cde_classifier", **train_kwargs)


def from_module() -> BaseClassifier:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_cde_classifier_plugin_sanity(test_plugin: BaseClassifier) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "cde_classifier"
    assert test_plugin.fqn() == "classification.cde_classifier"
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_cde_classifier_plugin_fit(test_plugin: BaseClassifier) -> None:
    dataset = SineDataloader().load()

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_cde_classifier_plugin_predict(test_plugin: BaseClassifier) -> None:
    dataset = SineDataloader().load()

    output = test_plugin.fit(dataset).predict(dataset)
    assert output.numpy().shape == (len(dataset.time_series), 1)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_cde_classifier_plugin_predict_proba(test_plugin: BaseClassifier) -> None:
    dataset = SineDataloader().load()

    output = test_plugin.fit(dataset).predict_proba(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 2)


def test_hyperparam_sample():
    for repeat in range(100):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
