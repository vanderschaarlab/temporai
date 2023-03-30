import pytest
from clairvoyance2.datasets.dummy import dummy_dataset

from tempor.data.clv2conv import clairvoyance2_dataset_to_tempor_dataset
from tempor.plugins import plugin_loader
from tempor.plugins.treatments import BaseTreatments
from tempor.plugins.treatments.plugin_crn_regressor import (
    CRNTreatmentsRegressor as plugin,
)
from tempor.utils.serialization import load, save

from .helpers_treatments import simulate_horizons, simulate_treatments_scenarios


def get_dummy_data(
    n_samples: int = 100,
    temporal_covariates_n_features: int = 5,
    temporal_covariates_max_len: int = 11,
    temporal_covariates_missing_prob: float = 0.0,
    static_covariates_n_features: int = 13,
    temporal_treatments_n_features: int = 5,
    temporal_treatments_n_categories: int = 2,
    temporal_targets_n_features: int = 7,
    temporal_targets_n_categories: int = 4,
):
    local_dataset = dummy_dataset(
        n_samples=n_samples,
        temporal_covariates_n_features=temporal_covariates_n_features,
        temporal_covariates_max_len=temporal_covariates_max_len,
        temporal_covariates_missing_prob=temporal_covariates_missing_prob,
        static_covariates_n_features=static_covariates_n_features,
        temporal_treatments_n_features=temporal_treatments_n_features,
        temporal_treatments_n_categories=temporal_treatments_n_categories,
        temporal_targets_n_features=temporal_targets_n_features,
        temporal_targets_n_categories=temporal_targets_n_categories,
        random_seed=12345,
    )
    return clairvoyance2_dataset_to_tempor_dataset(local_dataset)


def from_api() -> BaseTreatments:
    return plugin_loader.get("treatments.crn_regressor", random_state=123, epochs=10)


def from_module() -> BaseTreatments:
    return plugin(random_state=123, epochs=10)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_crn_regressor_plugin_sanity(test_plugin: BaseTreatments) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "crn_regressor"
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_crn_regressor_plugin_fit(test_plugin: BaseTreatments) -> None:
    data = get_dummy_data()
    test_plugin.fit(data)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_crn_regressor_plugin_predict(test_plugin: BaseTreatments) -> None:
    data = get_dummy_data(temporal_targets_n_features=3)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(data)

    dump = save(reloaded)
    reloaded = load(dump)

    horizons = simulate_horizons(data)
    output = reloaded.predict(data, horizons=horizons)

    assert output.numpy().shape == (len(data.time_series), 6, 3)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_crn_regressor_plugin_predict_counterfactuals(test_plugin: BaseTreatments) -> None:
    data = get_dummy_data(temporal_targets_n_features=3)
    test_plugin.fit(data)

    n_counterfactuals_per_sample = 2
    horizons, treatment_scenarios = simulate_treatments_scenarios(
        data, n_counterfactuals_per_sample=n_counterfactuals_per_sample
    )

    output = test_plugin.predict_counterfactuals(data, horizons=horizons, treatment_scenarios=treatment_scenarios)

    assert len(output) == len(data)
    assert len(output[0]) == n_counterfactuals_per_sample


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
