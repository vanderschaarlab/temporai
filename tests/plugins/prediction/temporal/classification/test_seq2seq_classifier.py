# pylint: disable=redefined-outer-name

from typing import Callable, Dict
from unittest.mock import Mock

import pytest

from tempor.plugins.prediction.temporal.classification import BaseTemporalClassifier
from tempor.plugins.prediction.temporal.classification.plugin_seq2seq_classifier import Seq2seqClassifier
from tempor.utils.serialization import load, save

INIT_KWARGS = {"random_state": 123, "epochs": 5}
PLUGIN_FROM_OPTIONS = ["from_api", pytest.param("from_module", marks=pytest.mark.extra)]
DEVICES = [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
TEST_ON_DATASETS = [
    "sine_data_temporal_small",
    pytest.param("sine_data_temporal_full", marks=pytest.mark.extra),
]


@pytest.fixture
def get_test_plugin(get_plugin: Callable):
    def func(plugin_from: str, base_kwargs: Dict, device: str):
        base_kwargs["device"] = device
        return get_plugin(
            plugin_from,
            fqn="prediction.temporal.classification.seq2seq_classifier",
            cls=Seq2seqClassifier,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS, device="cpu")
    assert test_plugin is not None
    assert test_plugin.name == "seq2seq_classifier"
    assert len(test_plugin.hyperparameter_space()) == 8


def test_fit_first_runtime_error(get_test_plugin: Callable, monkeypatch):
    from tempor.data.dataset import TemporalPredictionDataset

    test_plugin = get_test_plugin("from_api", INIT_KWARGS, device="cpu")
    monkeypatch.setattr(test_plugin, "_fit", Mock())

    test_plugin.fit(Mock(TemporalPredictionDataset))
    test_plugin.model = None

    with pytest.raises(RuntimeError, match=".*[Ff]it.*first.*"):
        test_plugin.predict(Mock(TemporalPredictionDataset), n_future_steps=1)


def test_predict_proba_not_implemented(get_test_plugin: Callable, monkeypatch):
    from tempor.data.dataset import TemporalPredictionDataset

    test_plugin = get_test_plugin("from_api", INIT_KWARGS, device="cpu")
    monkeypatch.setattr(test_plugin, "_fit", Mock())

    test_plugin.fit(Mock(TemporalPredictionDataset))
    test_plugin.model = None

    with pytest.raises(NotImplementedError):
        test_plugin.predict_proba(Mock(TemporalPredictionDataset), n_future_steps=1)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_fit(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseTemporalClassifier = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_predict(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseTemporalClassifier = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)

    test_plugin.fit(dataset)
    output = test_plugin.predict(dataset, n_future_steps=10)

    assert output.numpy().shape == (len(dataset.time_series), 10, 5)


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_serde(data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseTemporalClassifier = get_test_plugin("from_api", INIT_KWARGS, device=device)
    dataset = get_dataset(data)

    dump = save(test_plugin)
    reloaded1 = load(dump)

    reloaded1.fit(dataset)

    dump = save(reloaded1)
    reloaded2 = load(dump)

    reloaded2.predict(dataset, n_future_steps=10)
