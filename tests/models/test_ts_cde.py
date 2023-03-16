import math
from typing import Any

import numpy as np
import pytest
import torch

from tempor.models.ts_cde import NeuralCDE
from tempor.utils.datasets.google_stocks import GoogleStocksDataloader
from tempor.utils.datasets.sine import SineDataloader


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


def generate_dummy_data():
    seq_len = 100
    t = torch.linspace(0.0, 4 * math.pi, seq_len)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)

    temporal = X[perm]
    outcome = y[perm]
    observation_times = t.unsqueeze(0).repeat(len(X), 1)

    static = torch.randn(len(X), 3)
    return static, temporal, observation_times, outcome


def test_ode_sanity():
    static, temporal, observation_times, outcome = generate_dummy_data()

    model = NeuralCDE(
        task_type="classification",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=[2],
        n_units_hidden=8,
        n_iter=10,
    )
    model.fit(static, temporal, observation_times, outcome)

    # testing
    static, temporal, observation_times, outcome = generate_dummy_data()
    score = model.score(static, temporal, observation_times, outcome)
    print(score)
    assert score > 0


@pytest.mark.parametrize("source", [GoogleStocksDataloader, SineDataloader])
def test_ode_regression_fit_predict(source: Any) -> None:
    static, temporal, observation_times, outcome = unpack_dataset(source)

    outcome = outcome.reshape(-1, 1)

    outlen = int(len(outcome.reshape(-1)) / len(outcome))

    model = NeuralCDE(
        task_type="regression",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=outcome.shape[1:],
        n_iter=10,
        nonlin_out=[("tanh", outlen)],
    )

    model.fit(static, temporal, observation_times, outcome)

    y_pred = model.predict(static, temporal, observation_times)

    assert y_pred.shape == outcome.shape

    assert model.score(static, temporal, observation_times, outcome) < 2


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ode_classification_fit_predict(source: Any) -> None:
    static, temporal, observation_times, outcome = unpack_dataset(source)  # pylint: disable=unused-variable
    static_fake, temporal_fake = np.random.randn(*static.shape), np.random.randn(*temporal.shape)

    y = np.asarray([1] * len(static) + [0] * len(static_fake))  # type: ignore

    model = NeuralCDE(
        task_type="classification",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=[2],
        n_iter=10,
    )

    static_data = np.concatenate([static, static_fake])
    temporal_data = np.concatenate([temporal, temporal_fake])
    observation_times = np.concatenate([observation_times, observation_times])

    model.fit(static_data, temporal_data, observation_times, y)

    y_pred = model.predict(static_data, temporal_data, observation_times)

    assert y_pred.shape == y.shape

    print(model.score(static_data, temporal_data, observation_times, y))
    assert model.score(static_data, temporal_data, observation_times, y) <= 1
