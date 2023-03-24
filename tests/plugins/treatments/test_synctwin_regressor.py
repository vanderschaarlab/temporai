import pytest
from clairvoyance2.datasets import simple_pkpd_dataset

from tempor.data.clv2conv import clairvoyance2_dataset_to_tempor_dataset
from tempor.plugins import plugin_loader
from tempor.plugins.treatments import BaseTreatments
from tempor.plugins.treatments.plugin_synctwin_regressor import (
    SyncTwinTreatmentsRegressor as plugin,
)

from .helpers_treatments import simulate_treatments_scenarios


def get_dummy_data(
    n_timesteps: int = 30,
    time_index_treatment_event: int = 25,
    n_control_samples: int = 200,
    n_treated_samples: int = 200,
    seed: int = 100,
):
    local_dataset = simple_pkpd_dataset(
        n_timesteps=n_timesteps,
        time_index_treatment_event=time_index_treatment_event,
        n_control_samples=n_control_samples,
        n_treated_samples=n_treated_samples,
        seed=seed,
    )
    return clairvoyance2_dataset_to_tempor_dataset(local_dataset)


kwargs = {
    "pretraining_iterations": 10,
    "matching_iterations": 10,
    "inference_iterations": 10,
}


def from_api() -> BaseTreatments:
    return plugin_loader.get("treatments.synctwin_regressor", **kwargs)


def from_module() -> BaseTreatments:
    return plugin(**kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_synctwin_regressor_plugin_sanity(test_plugin: BaseTreatments) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "synctwin_regressor"
    assert len(test_plugin.hyperparameter_space()) == 5


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_synctwin_regressor_plugin_fit(test_plugin: BaseTreatments) -> None:
    data = get_dummy_data()
    test_plugin.fit(data)


# TODO: Handle this scenario.
@pytest.mark.xfail
@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_synctwin_regressor_plugin_predict_counterfactuals(test_plugin: BaseTreatments) -> None:
    data = get_dummy_data()
    test_plugin.fit(data)

    n_counterfactuals_per_sample = 2
    horizons, treatment_scenarios = simulate_treatments_scenarios(
        data, n_counterfactuals_per_sample=n_counterfactuals_per_sample
    )

    output = test_plugin.predict_counterfactuals(data, horizons=horizons, treatment_scenarios=treatment_scenarios)

    assert len(output) == len(data)


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
