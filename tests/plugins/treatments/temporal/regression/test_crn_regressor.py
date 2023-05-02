import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.treatments.temporal import BaseTemporalTreatmentEffects
from tempor.plugins.treatments.temporal.regression.plugin_crn_regressor import CRNTreatmentsRegressor as plugin
from tempor.utils.serialization import load, save

from ...helpers_treatments import simulate_horizons, simulate_treatments_scenarios

train_kwargs = {"random_state": 123, "n_iter": 3}

TEST_ON_DATASETS = ["clv_data_small"]


def from_api() -> BaseTemporalTreatmentEffects:
    return plugin_loader.get("treatments.temporal.regression.crn_regressor", **train_kwargs)


def from_module() -> BaseTemporalTreatmentEffects:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseTemporalTreatmentEffects) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "crn_regressor"
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseTemporalTreatmentEffects, data: str, request: pytest.FixtureRequest) -> None:
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
def test_predict(test_plugin: BaseTemporalTreatmentEffects, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    horizons = simulate_horizons(dataset)
    output = reloaded.predict(dataset, horizons=horizons)

    assert output.numpy().shape == (len(dataset.time_series), 6, 3)


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_predict_counterfactuals(
    test_plugin: BaseTemporalTreatmentEffects, data: str, request: pytest.FixtureRequest
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
