from typing import Type

import pytest

from tempor.data.dataloader import DataLoader
from tempor.data.dataset import PredictiveDataset
from tempor.utils import dataloaders


@pytest.mark.parametrize(
    "dataloader_cls",
    [
        dataloaders.SineDataLoader,
        dataloaders.GoogleStocksDataLoader,
        dataloaders.GoogleStocksDataLoader,
        dataloaders.PKPDDataLoader,
        dataloaders.DummyTemporalPredictionDataLoader,
        dataloaders.DummyTemporalTreatmentEffectsDataLoader,
    ],
)
def test_init_load_and_basic_methods(dataloader_cls: Type[DataLoader]):
    dataloader = dataloader_cls()  # Test __ini__ with no (all default) parameters.

    url = dataloader.url()
    dataset_dir = dataloader.dataset_dir()
    dataset = dataloader.load()

    assert isinstance(url, (str, type(None)))
    assert isinstance(dataset_dir, (str, type(None)))
    assert isinstance(dataset, PredictiveDataset)

    # Assert the right type of dataset for the dataloader has been returned.
    return_type = dataloader_cls.load.__annotations__["return"]
    assert isinstance(dataset, return_type)
