import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.treatments import BaseTreatments
from tempor.plugins.treatments.plugin_crn_classifier import (
    CRNTreatmentsClassifier as plugin,
)
from tempor.utils.serialization import load, save

from .helpers_treatments import simulate_horizons, simulate_treatments_scenarios

train_kwargs = {"random_state": 123, "n_iter": 3}

TEST_ON_DATASETS = ["clv_data_small"]


def from_api() -> BaseTreatments:
    return plugin_loader.get("treatments.crn_classifier", **train_kwargs)


def from_module() -> BaseTreatments:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseTreatments) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "crn_classifier"
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseTreatments, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_predict(test_plugin: BaseTreatments, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    horizons = simulate_horizons(dataset)
    output = reloaded.predict(dataset, horizons=horizons)

    assert output.numpy().shape == (len(dataset.time_series), 6, 3)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_crn_classifier_plugin_predict_counterfactuals(
    test_plugin: BaseTreatments, data: str, request: pytest.FixtureRequest
) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)

    n_counterfactuals_per_sample = 2
    horizons, treatment_scenarios = simulate_treatments_scenarios(
        dataset, n_counterfactuals_per_sample=n_counterfactuals_per_sample
    )

    output = test_plugin.predict_counterfactuals(dataset, horizons=horizons, treatment_scenarios=treatment_scenarios)

    assert len(output) == len(dataset)
    assert len(output[0]) == n_counterfactuals_per_sample
