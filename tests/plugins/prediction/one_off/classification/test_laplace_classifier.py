# pylint: disable=redefined-outer-name

from typing import Callable, Dict
from unittest.mock import Mock

import pytest

from tempor.plugins.prediction.one_off.classification import BaseOneOffClassifier
from tempor.plugins.prediction.one_off.classification.plugin_laplace_classifier import LaplaceODEClassifier
from tempor.utils.serialization import load, save

INIT_KWARGS = {"random_state": 123, "n_iter": 5}
PLUGIN_FROM_OPTIONS = ["from_api", pytest.param("from_module", marks=pytest.mark.extra)]
DEVICES = [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
TEST_ON_DATASETS = [
    "google_stocks_data_small",
    pytest.param("google_stocks_data_full", marks=pytest.mark.extra),
]


@pytest.fixture
def get_test_plugin(get_plugin: Callable):
    def func(plugin_from: str, base_kwargs: Dict, device: str):
        base_kwargs["device"] = device
        return get_plugin(
            plugin_from,
            fqn="prediction.one_off.classification.laplace_ode_classifier",
            cls=LaplaceODEClassifier,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS, device="cpu")
    assert test_plugin is not None
    assert test_plugin.name == "laplace_ode_classifier"
    assert test_plugin.fqn() == "prediction.one_off.classification.laplace_ode_classifier"
    assert len(test_plugin.hyperparameter_space()) == 7


def test_fit_first_runtime_error(get_test_plugin: Callable, monkeypatch):
    from tempor.data.dataset import OneOffPredictionDataset

    test_plugin = get_test_plugin("from_api", INIT_KWARGS, device="cpu")
    monkeypatch.setattr(test_plugin, "_fit", Mock())

    test_plugin.fit(Mock(OneOffPredictionDataset))
    test_plugin.model = None

    with pytest.raises(RuntimeError, match=".*[Ff]it.*first.*"):
        test_plugin.predict(Mock(OneOffPredictionDataset))

    with pytest.raises(RuntimeError, match=".*[Ff]it.*first.*"):
        test_plugin.predict_proba(Mock(OneOffPredictionDataset))


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_fit(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseOneOffClassifier = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "no_targets",
    [
        False,
        pytest.param(True, marks=pytest.mark.extra),
    ],
)
def test_predict(
    plugin_from: str, data: str, device: str, no_targets: bool, get_test_plugin: Callable, get_dataset: Callable
) -> None:
    test_plugin: BaseOneOffClassifier = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)
    if no_targets:
        dataset.predictive.targets = None
    output = test_plugin.predict(dataset)
    assert output.numpy().shape == (len(dataset.time_series), 1)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_predict_proba(
    plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable
) -> None:
    test_plugin: BaseOneOffClassifier = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)

    output = test_plugin.fit(dataset).predict_proba(dataset)

    if data == "google_stocks_data_small":
        out_classes = 6
    elif data == "google_stocks_data_full":
        out_classes = 10
    else:
        raise NotImplementedError
    assert output.numpy().shape == (len(dataset.time_series), out_classes)


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_serde(data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseOneOffClassifier = get_test_plugin("from_api", INIT_KWARGS, device=device)
    dataset = get_dataset(data)

    dump = save(test_plugin)
    reloaded1 = load(dump)

    reloaded1.fit(dataset)

    dump = save(reloaded1)
    reloaded2 = load(dump)

    reloaded2.predict(dataset)
