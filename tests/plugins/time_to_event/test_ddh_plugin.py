from typing import TYPE_CHECKING, Callable

import pytest

from tempor.benchmarks.evaluation import evaluate_time_to_event
from tempor.plugins import plugin_loader
from tempor.plugins.time_to_event.plugin_ddh import DynamicDeepHitTimeToEventAnalysis as plugin
from tempor.utils.serialization import load, save

if TYPE_CHECKING:  # pragma: no cover
    from tempor.plugins.time_to_event import BaseTimeToEventAnalysis

train_kwargs = {"random_state": 123, "n_iter": 5}

TEST_ON_DATASETS = ["pbc_data_small"]


def from_api() -> "BaseTimeToEventAnalysis":
    return plugin_loader.get("time_to_event.dynamic_deephit", **train_kwargs)


def from_module() -> "BaseTimeToEventAnalysis":
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: "BaseTimeToEventAnalysis") -> None:
    assert test_plugin is not None
    assert test_plugin.name == "dynamic_deephit"
    assert test_plugin.fqn() == "time_to_event.dynamic_deephit"
    assert len(test_plugin.hyperparameter_space()) == 10


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: "BaseTimeToEventAnalysis", data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_predict(
    test_plugin: "BaseTimeToEventAnalysis",
    get_event0_time_percentiles: Callable,
    data: str,
    request: pytest.FixtureRequest,
) -> None:
    dataset = request.getfixturevalue(data)

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.predict(dataset, horizons=horizons)

    assert output.numpy().shape == (len(dataset.time_series), len(horizons), 1)


@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_benchmark(get_event0_time_percentiles: Callable, data: str, request: pytest.FixtureRequest) -> None:
    test_plugin = from_api()
    dataset = request.getfixturevalue(data)

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    score = evaluate_time_to_event(test_plugin, dataset, horizons)

    assert (score.loc[:, "errors"] == 0).all()
    assert score.loc["c_index", "mean"] > 0.5  # pyright: ignore
