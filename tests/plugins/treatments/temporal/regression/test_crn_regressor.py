# pylint: disable=redefined-outer-name

from typing import TYPE_CHECKING, Callable, Dict

import pytest

from tempor.plugins.treatments.temporal.regression.plugin_crn_regressor import CRNTreatmentsRegressor
from tempor.utils.serialization import load, save

from ...helpers_treatments import simulate_horizons, simulate_treatments_scenarios

if TYPE_CHECKING:  # pragma: no cover
    from tempor.plugins.treatments.temporal import BaseTemporalTreatmentEffects

INIT_KWARGS = {"random_state": 123, "n_iter": 3}
PLUGIN_FROM_OPTIONS = ["from_api", pytest.param("from_module", marks=pytest.mark.extra)]
DEVICES = [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
TEST_ON_DATASETS = [
    "clv_data_small",
    pytest.param("clv_data_full", marks=pytest.mark.extra),
]


@pytest.fixture
def get_test_plugin(get_plugin: Callable):
    def func(plugin_from: str, base_kwargs: Dict, device: str):
        base_kwargs["device"] = device
        return get_plugin(
            plugin_from,
            fqn="treatments.temporal.regression.crn_regressor",
            cls=CRNTreatmentsRegressor,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS, device="cpu")
    assert test_plugin is not None
    assert test_plugin.name == "crn_regressor"
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_fit(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: "BaseTemporalTreatmentEffects" = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_predict(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: "BaseTemporalTreatmentEffects" = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    horizons = simulate_horizons(dataset)
    output = reloaded.predict(dataset, horizons=horizons)

    assert output.numpy().shape == (len(dataset.time_series), 6, 3)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_predict_counterfactuals(
    plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable
) -> None:
    test_plugin: "BaseTemporalTreatmentEffects" = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)

    n_counterfactuals_per_sample = 2
    horizons, treatment_scenarios = simulate_treatments_scenarios(
        dataset, n_counterfactuals_per_sample=n_counterfactuals_per_sample
    )

    output = test_plugin.predict_counterfactuals(dataset, horizons=horizons, treatment_scenarios=treatment_scenarios)

    assert len(output) == len(dataset)
    assert len(output[0]) == n_counterfactuals_per_sample
