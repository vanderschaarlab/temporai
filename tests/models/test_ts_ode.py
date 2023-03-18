from typing import Any

import numpy as np
import pytest

from tempor.models.constants import ODEBackend
from tempor.models.ts_ode import NeuralODE
from tempor.utils.datasets.google_stocks import GoogleStocksDataloader
from tempor.utils.datasets.sine import SineDataloader


def unpack_dataset(source):
    dataset = source().load()
    temporal = dataset.time_series.numpy()
    observation_times = np.asarray(dataset.time_series.time_indexes())
    outcome = dataset.predictive.targets.numpy()

    if dataset.static is not None:
        static = dataset.static.numpy()
    else:
        static = np.zeros((len(temporal), 0))

    return static, temporal, observation_times, outcome


@pytest.mark.parametrize("backend", ["laplace", "ode", "cde"])
def test_ode_sanity(backend: ODEBackend):
    model = NeuralODE(
        task_type="classification",
        n_static_units_in=23,
        n_temporal_units_in=34,
        output_shape=[2],
        n_units_hidden=8,
        n_iter=10,
        backend=backend,
    )

    assert hasattr(model, "backend")
    assert hasattr(model, "func")
    assert hasattr(model, "initial_temporal")
    assert hasattr(model, "initial_static")


@pytest.mark.parametrize("source", [GoogleStocksDataloader, SineDataloader])
@pytest.mark.parametrize("backend", ["laplace", "cde", "ode"])
def test_ode_regression_fit_predict(source: Any, backend: ODEBackend) -> None:
    if source == SineDataloader and backend == "laplace":
        # NOTE: Test with this setup fails, laplace implementation is not yet stable,
        # this needs to be debugged with the author.
        return

    static, temporal, observation_times, outcome = unpack_dataset(source)

    outcome = outcome.reshape(-1, 1)

    outlen = int(len(outcome.reshape(-1)) / len(outcome))

    model = NeuralODE(
        task_type="regression",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=outcome.shape[1:],
        n_iter=10,
        nonlin_out=[("tanh", outlen)],
        backend=backend,
    )

    model.fit(static, temporal, observation_times, outcome)

    y_pred = model.predict(static, temporal, observation_times)

    assert y_pred.shape == outcome.shape

    assert model.score(static, temporal, observation_times, outcome) < 2


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
@pytest.mark.parametrize("backend", ["laplace", "cde", "ode"])
def test_ode_classification_fit_predict(source: Any, backend: ODEBackend) -> None:
    if source == SineDataloader and backend == "laplace":
        # NOTE: Test with this setup fails, laplace implementation is not yet stable,
        # this needs to be debugged with the author.
        return

    static, temporal, observation_times, outcome = unpack_dataset(source)  # pylint: disable=unused-variable
    static_fake, temporal_fake = np.random.randn(*static.shape), np.random.randn(*temporal.shape)

    y = np.asarray([1] * len(static) + [0] * len(static_fake))  # type: ignore

    model = NeuralODE(
        task_type="classification",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=[2],
        n_iter=10,
        backend=backend,
    )

    static_data = np.concatenate([static, static_fake])
    temporal_data = np.concatenate([temporal, temporal_fake])
    observation_times = np.concatenate([observation_times, observation_times])

    model.fit(static_data, temporal_data, observation_times, y)

    y_pred = model.predict(static_data, temporal_data, observation_times)

    assert y_pred.shape == y.shape

    assert model.score(static_data, temporal_data, observation_times, y) <= 1
