from typing import TYPE_CHECKING, Callable

import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.time_to_event.plugin_ddh import (
    DynamicDeepHitSurvivalAnalysis as plugin,
)
from tempor.utils.dataloaders import PBCDataLoader

if TYPE_CHECKING:  # pragma: no cover
    from tempor.plugins.time_to_event import BaseTimeToEventAnalysis

train_kwargs = {"random_state": 123, "n_iter": 10}


def from_api() -> "BaseTimeToEventAnalysis":
    return plugin_loader.get("time_to_event.dynamic_deephit", **train_kwargs)


def from_module() -> "BaseTimeToEventAnalysis":
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_dynamic_deephit_plugin_sanity(test_plugin: "BaseTimeToEventAnalysis") -> None:
    assert test_plugin is not None
    assert test_plugin.name == "dynamic_deephit"
    assert test_plugin.fqn() == "time_to_event.dynamic_deephit"
    assert len(test_plugin.hyperparameter_space()) == 10


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_dynamic_deephit_plugin_fit(test_plugin: "BaseTimeToEventAnalysis") -> None:
    dataset = PBCDataLoader().load()
    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_dynamic_deephit_plugin_predict(
    test_plugin: "BaseTimeToEventAnalysis", get_event0_time_percentiles: Callable
) -> None:
    dataset = PBCDataLoader().load()

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])

    output = test_plugin.fit(dataset).predict(dataset, horizons=horizons)

    assert output.numpy().shape == (len(dataset.time_series), len(horizons), 1)


def test_hyperparam_sample():
    for repeat in range(100):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
