import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.classification import BaseClassifier
from tempor.plugins.classification.plugin_ode_classifier import ODEClassifier as plugin
from tempor.utils.dataloaders.sine import SineDataLoader

train_kwargs = {"random_state": 123, "n_iter": 50}


def from_api() -> BaseClassifier:
    return plugin_loader.get("classification.ode_classifier", **train_kwargs)


def from_module() -> BaseClassifier:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ode_classifier_plugin_sanity(test_plugin: BaseClassifier) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "ode_classifier"
    assert test_plugin.fqn() == "classification.ode_classifier"
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ode_classifier_plugin_fit(test_plugin: BaseClassifier) -> None:
    dataset = SineDataLoader().load()

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ode_classifier_plugin_predict(test_plugin: BaseClassifier) -> None:
    dataset = SineDataLoader().load()

    output = test_plugin.fit(dataset).predict(dataset)
    assert output.numpy().shape == (len(dataset.time_series), 1)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ode_classifier_plugin_predict_proba(test_plugin: BaseClassifier) -> None:
    dataset = SineDataLoader().load()

    output = test_plugin.fit(dataset).predict_proba(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 2)


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
