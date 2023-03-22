import numpy as np
import pytest
import torch

from tempor.models.constants import DEVICE
from tempor.models.transformer import TransformerModel
from tempor.utils.dataloaders import GoogleStocksDataLoader, SineDataLoader


def unpack_dataset(source):
    dataset = source().load()
    temporal = dataset.time_series.numpy()
    observation_times = dataset.time_series.time_indexes()
    outcome = dataset.predictive.targets.numpy()

    if dataset.static is not None:
        static = dataset.static.numpy()
    else:
        static = np.zeros((len(temporal), 0))

    return static, temporal, observation_times, outcome


@pytest.mark.parametrize("source", [GoogleStocksDataLoader, SineDataLoader])
def test_sanity(source) -> None:
    _, temporal, _, _ = unpack_dataset(source)  # pylint: disable=unused-variable

    n_hidden = 10
    model = TransformerModel(n_units_in=temporal[0].shape[-1], n_units_hidden=n_hidden)
    temporal = torch.from_numpy(temporal).to(device=DEVICE, dtype=torch.float)
    out = model.forward(temporal)

    assert out.shape == (len(temporal), temporal[0].shape[0], n_hidden)
