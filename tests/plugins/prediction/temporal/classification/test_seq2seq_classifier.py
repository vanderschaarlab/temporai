import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.prediction.temporal.classification import BaseTemporalClassifier
from tempor.plugins.prediction.temporal.classification.plugin_seq2seq_classifier import (
    Seq2seqClassifier as plugin,
)
from tempor.utils.serialization import load, save

train_kwargs = {"random_state": 123, "epochs": 5}

TEST_ON_DATASETS = ["sine_data_temporal_small"]


def from_api() -> BaseTemporalClassifier:
    return plugin_loader.get("prediction.temporal.classification.seq2seq_classifier", **train_kwargs)


def from_module() -> BaseTemporalClassifier:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseTemporalClassifier) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "seq2seq_classifier"
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseTemporalClassifier, data: str, request: pytest.FixtureRequest) -> None:
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
def test_predict(test_plugin: BaseTemporalClassifier, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)
    output = test_plugin.predict(dataset, n_future_steps=10)

    assert output.numpy().shape == (len(dataset.time_series), 10, 5)


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_serde(data: str, request: pytest.FixtureRequest) -> None:
    test_plugin = from_api()

    dataset = request.getfixturevalue(data)

    dump = save(test_plugin)
    reloaded1 = load(dump)

    reloaded1.fit(dataset)

    dump = save(reloaded1)
    reloaded2 = load(dump)

    reloaded2.predict(dataset, n_future_steps=10)
