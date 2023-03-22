import pytest
from clairvoyance2.datasets.dummy import dummy_dataset

from tempor.data.clv2conv import clairvoyance2_dataset_to_tempor_dataset
from tempor.plugins import plugin_loader
from tempor.plugins.treatments import BaseTreatments
from tempor.plugins.treatments.plugin_crn_regressor import (
    CRNTreatmentsRegressor as plugin,
)


def get_dummy_data(temporal_dim: int = 5):
    local_dataset = dummy_dataset(
        n_samples=100,
        temporal_covariates_n_features=temporal_dim,
        temporal_covariates_max_len=6 * 2,
        temporal_covariates_missing_prob=0.0,
        static_covariates_n_features=0,
        temporal_treatments_n_features=2,
        temporal_treatments_n_categories=1,
        random_seed=12345,
    )
    print(local_dataset)
    return clairvoyance2_dataset_to_tempor_dataset(local_dataset)


def from_api() -> BaseTreatments:
    return plugin_loader.get("treatments.crn_regressor", random_state=123, epochs=10)


def from_module() -> BaseTreatments:
    return plugin(random_state=123, epochs=10)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_CRN_regressor_plugin_sanity(test_plugin: BaseTreatments) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "crn_regressor"
    assert len(test_plugin.hyperparameter_space()) == 6


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_CRN_regressor_plugin_fit(test_plugin: BaseTreatments) -> None:
    data = get_dummy_data()
    test_plugin.fit(data)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_CRN_regressor_plugin_predict(test_plugin: BaseTreatments) -> None:
    temporal_dim = 5
    data = get_dummy_data(temporal_dim=temporal_dim)
    test_plugin.fit(data)
    output = test_plugin.predict(data, n_future_steps=10)

    assert output.numpy().shape == (len(data.time_series), 10, temporal_dim)


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
