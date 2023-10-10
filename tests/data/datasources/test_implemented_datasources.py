from typing import Type

import pytest

from tempor.data.dataset import PredictiveDataset
from tempor.data.datasources.datasource import DataSource


def from_modules():
    # Import by module.
    from tempor.data.datasources.prediction.one_off.plugin_google_stocks import GoogleStocksDataSource
    from tempor.data.datasources.prediction.one_off.plugin_sine import SineDataSource
    from tempor.data.datasources.prediction.temporal.plugin_dummy_prediction import DummyTemporalPredictionDataSource
    from tempor.data.datasources.prediction.temporal.plugin_uci_diabetes import UCIDiabetesDataSource
    from tempor.data.datasources.time_to_event.plugin_pbc import PBCDataSource
    from tempor.data.datasources.treatments.one_off.plugin_pkpd import PKPDDataSource
    from tempor.data.datasources.treatments.temporal.plugin_dummy_treatments import (
        DummyTemporalTreatmentEffectsDataSource,
    )

    return (
        GoogleStocksDataSource,
        SineDataSource,
        DummyTemporalPredictionDataSource,
        UCIDiabetesDataSource,
        PBCDataSource,
        PKPDDataSource,
        DummyTemporalTreatmentEffectsDataSource,
    )


# TODO: Import by plugin API.


REQUIRE_DOWNLOAD = ["google_stocks", "pbc", "uci_diabetes"]


@pytest.mark.parametrize("datasource_cls", from_modules())
def test_init_load_and_basic_methods(datasource_cls: Type[DataSource]):
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


@pytest.mark.parametrize("datasource_cls", [x for x in from_modules() if x.name in REQUIRE_DOWNLOAD])
def test_download_and_local(datasource_cls, tmpdir, monkeypatch):
    temp_data_root_dir = tmpdir.mkdir("data_root")
    monkeypatch.setattr(DataSource, "data_root_dir", str(temp_data_root_dir))

    dl = datasource_cls()
    # First time - download.
    dl.load()
    # Second time - local.
    dl.load()
