from typing import Type

import pytest

from tempor import plugin_loader
from tempor.data.dataset import PredictiveDataset
from tempor.datasources.datasource import DataSource

# Importing directly from modules.
from tempor.datasources.prediction.one_off.plugin_google_stocks import GoogleStocksDataSource
from tempor.datasources.prediction.one_off.plugin_sine import SineDataSource
from tempor.datasources.prediction.temporal.plugin_dummy_prediction import DummyTemporalPredictionDataSource
from tempor.datasources.prediction.temporal.plugin_uci_diabetes import UCIDiabetesDataSource
from tempor.datasources.time_to_event.plugin_pbc import PBCDataSource
from tempor.datasources.treatments.one_off.plugin_pkpd import PKPDDataSource
from tempor.datasources.treatments.temporal.plugin_dummy_treatments import DummyTemporalTreatmentEffectsDataSource

# DEFINITIONS FOR TESTS ---

# Keys: plugin full name
# Values: plugin class.
PLUGINS_TO_TEST = {
    "prediction.one_off.google_stocks": GoogleStocksDataSource,
    "prediction.one_off.sine": SineDataSource,
    "prediction.temporal.dummy_prediction": DummyTemporalPredictionDataSource,
    "prediction.temporal.uci_diabetes": UCIDiabetesDataSource,
    "time_to_event.pbc": PBCDataSource,
    "treatments.one_off.pkpd": PKPDDataSource,
    "treatments.temporal.dummy_treatments": DummyTemporalTreatmentEffectsDataSource,
}

# List the plugins that require downloading from the internet.
REQUIRE_DOWNLOAD = ["prediction.one_off.google_stocks", "time_to_event.pbc", "prediction.temporal.uci_diabetes"]

# Sanity check for the set of tests - we want tests to fail if there is mismatch between the lists.
for x in REQUIRE_DOWNLOAD:
    if x not in PLUGINS_TO_TEST:
        raise ValueError(f"Plugin {x} is in REQUIRE_DOWNLOAD but not in PLUGINS_TO_TEST.")

# DEFINITIONS FOR TESTS (END) ---


def get_plugin(plugin_from: str, plugin_full_name: str) -> Type[DataSource]:
    if plugin_from == "module":
        return PLUGINS_TO_TEST[plugin_full_name]
    else:
        return plugin_loader.get_class(plugin_full_name, plugin_type="datasource")


@pytest.mark.parametrize("plugin_from", ["module", "api"])
@pytest.mark.parametrize("plugin_full_name", PLUGINS_TO_TEST.keys())
def test_init_load_and_basic_methods(plugin_from: str, plugin_full_name: str):
    datasource_cls = get_plugin(plugin_from, plugin_full_name)
    datasource = datasource_cls()  # Test __ini__ with no (all default) parameters.

    url = datasource.url()
    dataset_dir = datasource.dataset_dir()
    dataset = datasource.load()

    assert isinstance(url, (str, type(None)))
    assert isinstance(dataset_dir, (str, type(None)))
    assert isinstance(dataset, PredictiveDataset)

    # Assert the right type of dataset for the datasource has been returned.
    return_type = datasource_cls.load.__annotations__["return"]
    assert isinstance(dataset, return_type)


@pytest.mark.parametrize("plugin_from", ["module", "api"])
@pytest.mark.parametrize("plugin_full_name", [x for x in PLUGINS_TO_TEST.keys() if x in REQUIRE_DOWNLOAD])
def test_download_and_local(plugin_from: str, plugin_full_name: str, tmpdir, monkeypatch):
    temp_data_root_dir = tmpdir.mkdir("data_root")
    monkeypatch.setattr(DataSource, "data_root_dir", str(temp_data_root_dir))

    datasource_cls = get_plugin(plugin_from, plugin_full_name)
    datasource = datasource_cls()
    # First time - download.
    datasource.load()
    # Second time - local.
    datasource.load()
