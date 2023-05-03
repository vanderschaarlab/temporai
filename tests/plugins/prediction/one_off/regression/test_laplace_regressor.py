# pylint: disable=redefined-outer-name

from typing import Callable, Dict

import pytest

from tempor.plugins.prediction.one_off.regression import BaseOneOffRegressor
from tempor.plugins.prediction.one_off.regression.plugin_laplace_regressor import LaplaceODERegressor
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
            fqn="prediction.one_off.regression.laplace_ode_regressor",
            cls=LaplaceODERegressor,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS, device="cpu")
    assert test_plugin is not None
    assert test_plugin.name == "laplace_ode_regressor"
    assert len(test_plugin.hyperparameter_space()) == 7


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_fit(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseOneOffRegressor = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
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
    test_plugin: BaseOneOffRegressor = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    if no_targets:
        dataset.predictive.targets = None

    output = reloaded.predict(dataset)

    assert output.numpy().shape == (len(dataset.time_series), 1)
