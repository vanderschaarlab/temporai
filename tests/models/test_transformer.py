import numpy as np
import pytest
import torch

from tempor.data.datasources.prediction.one_off.plugin_google_stocks import GoogleStocksDataSource
from tempor.data.datasources.prediction.one_off.plugin_sine import SineDataSource
from tempor.models.constants import DEVICE
from tempor.models.transformer import TransformerModel, Transpose


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


@pytest.mark.parametrize("source", [GoogleStocksDataSource, SineDataSource])
def test_sanity(source) -> None:
    _, temporal, _, _ = unpack_dataset(source)

    n_hidden = 10
    model = TransformerModel(n_units_in=temporal[0].shape[-1], n_units_hidden=n_hidden)
    temporal = torch.from_numpy(temporal).to(device=DEVICE, dtype=torch.float)
    out = model.forward(temporal)

    assert out.shape == (len(temporal), temporal[0].shape[0], n_hidden)


@pytest.mark.parametrize("contiguous", [True, False])
def test_transpose(contiguous):
    x = torch.ones(size=(2, 3))
    tr = Transpose(1, 0, contiguous=contiguous)
    out = tr(x)
    assert list(out.shape) == [3, 2]
