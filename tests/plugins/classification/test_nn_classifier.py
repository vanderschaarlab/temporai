import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.classification import BaseClassifier
from tempor.plugins.classification.plugin_nn_classifier import (
    NeuralNetClassifier as plugin,
)
from tempor.utils.serialization import load, save

train_kwargs = {"random_state": 123, "n_iter": 5}

TEST_ON_DATASETS = ["sine_data_small"]


def from_api() -> BaseClassifier:
    return plugin_loader.get("classification.nn_classifier", **train_kwargs)


def from_module() -> BaseClassifier:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseClassifier) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "nn_classifier"
    assert test_plugin.fqn() == "classification.nn_classifier"
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseClassifier, data: str, request: pytest.FixtureRequest) -> None:
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
def test_predict(test_plugin: BaseClassifier, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    output = test_plugin.fit(dataset).predict(dataset)
    assert output.numpy().shape == (len(dataset.time_series), 1)


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_predict_proba(test_plugin: BaseClassifier, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    output = test_plugin.fit(dataset).predict_proba(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 2)


@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_serde(data: str, request: pytest.FixtureRequest) -> None:
    test_plugin = from_api()

    dataset = request.getfixturevalue(data)

    dump = save(test_plugin)
    reloaded1 = load(dump)

    reloaded1.fit(dataset)

    dump = save(reloaded1)
    reloaded2 = load(dump)

    reloaded2.predict(dataset)
