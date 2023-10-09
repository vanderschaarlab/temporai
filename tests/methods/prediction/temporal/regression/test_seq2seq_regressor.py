# pylint: disable=redefined-outer-name

from typing import Callable, Dict
from unittest.mock import Mock

import pytest

from tempor.methods.prediction.temporal.regression import BaseTemporalRegressor
from tempor.methods.prediction.temporal.regression.plugin_seq2seq_regressor import Seq2seqRegressor
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
            fqn="prediction.temporal.regression.seq2seq_regressor",
            cls=Seq2seqRegressor,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS, device="cpu")
    assert test_plugin is not None
    assert test_plugin.name == "seq2seq_regressor"
    assert len(test_plugin.hyperparameter_space()) == 8


def test_fit_first_runtime_error(get_test_plugin: Callable, monkeypatch):
    from tempor.data.dataset import TemporalPredictionDataset

    test_plugin = get_test_plugin("from_api", INIT_KWARGS, device="cpu")
    monkeypatch.setattr(test_plugin, "_fit", Mock())

    test_plugin.fit(Mock(TemporalPredictionDataset))
    test_plugin.model = None

    with pytest.raises(RuntimeError, match=".*[Ff]it.*first.*"):
        test_plugin.predict(Mock(TemporalPredictionDataset), n_future_steps=1)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_fit(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseTemporalRegressor = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_predict(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseTemporalRegressor = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.predict(dataset, n_future_steps=10)

    assert output.numpy().shape == (len(dataset.time_series), 10, 5)
