# pylint: disable=redefined-outer-name

from typing import TYPE_CHECKING, Callable, Dict

import pytest

from tempor.benchmarks.evaluation import evaluate_time_to_event
from tempor.plugins.time_to_event.plugin_ddh import DynamicDeepHitTimeToEventAnalysis
from tempor.utils.serialization import load, save

if TYPE_CHECKING:  # pragma: no cover
    from tempor.plugins.time_to_event import BaseTimeToEventAnalysis

INIT_KWARGS = {"random_state": 123, "n_iter": 5}
PLUGIN_FROM_OPTIONS = ["from_api", pytest.param("from_module", marks=pytest.mark.extra)]
DEVICES = [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
TEST_ON_DATASETS = [
    "pbc_data_small",
    pytest.param("pbc_data_full", marks=pytest.mark.extra),
]


@pytest.fixture
def get_test_plugin(get_plugin: Callable):
    def func(plugin_from: str, base_kwargs: Dict, device: str):
        base_kwargs["device"] = device
        return get_plugin(
            plugin_from,
            fqn="time_to_event.dynamic_deephit",
            cls=DynamicDeepHitTimeToEventAnalysis,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin: "BaseTimeToEventAnalysis" = get_test_plugin(plugin_from, INIT_KWARGS, device="cpu")
    assert test_plugin is not None
    assert test_plugin.name == "dynamic_deephit"
    assert test_plugin.fqn() == "time_to_event.dynamic_deephit"
    assert len(test_plugin.hyperparameter_space()) == 10


@pytest.mark.filterwarnings("ignore:.*Validation.*small.*:UserWarning")  # Expected for small test datasets.
@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_fit(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: "BaseTimeToEventAnalysis" = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


@pytest.mark.filterwarnings("ignore:.*Validation.*small.*:UserWarning")  # Expected for small test datasets.
@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_predict(
    plugin_from: str,
    data: str,
    device: str,
    get_test_plugin: Callable,
    get_dataset: Callable,
    get_event0_time_percentiles: Callable,
) -> None:
    test_plugin: "BaseTimeToEventAnalysis" = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.predict(dataset, horizons=horizons)

    assert output.numpy().shape == (len(dataset.time_series), len(horizons), 1)


@pytest.mark.filterwarnings("ignore:.*Validation.*small.*:UserWarning")  # Expected for small test datasets.
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_benchmark(
    data: str, device: str, get_test_plugin: Callable, get_dataset: Callable, get_event0_time_percentiles: Callable
) -> None:
    test_plugin: "BaseTimeToEventAnalysis" = get_test_plugin("from_api", INIT_KWARGS, device=device)
    dataset = get_dataset(data)

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    score = evaluate_time_to_event(test_plugin, dataset, horizons)

    assert (score.loc[:, "errors"] == 0).all()
    assert score.loc["c_index", "mean"] > 0.5  # pyright: ignore
