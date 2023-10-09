from typing import Type

import pytest

from tempor.data import datasources
from tempor.data.dataset import PredictiveDataset
from tempor.data.datasources.datasource import DataSource


@pytest.mark.parametrize(
    "datasource_cls",
    [
        datasources.SineDataSource,
        datasources.GoogleStocksDataSource,
        datasources.PKPDDataSource,
        datasources.PBCDataSource,
        datasources.DummyTemporalPredictionDataSource,
        datasources.DummyTemporalTreatmentEffectsDataSource,
        datasources.UCIDiabetesDataSource,
    ],
)
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


@pytest.mark.parametrize(
    "datasource_cls",
    [
        datasources.GoogleStocksDataSource,
        datasources.PBCDataSource,
        datasources.UCIDiabetesDataSource,
    ],
)
def test_download_and_local(datasource_cls, tmpdir, monkeypatch):
    temp_data_root_dir = tmpdir.mkdir("data_root")
    monkeypatch.setattr(DataSource, "data_root_dir", str(temp_data_root_dir))

    dl = datasource_cls()
    # First time - download.
    dl.load()
    # Second time - local.
    dl.load()
