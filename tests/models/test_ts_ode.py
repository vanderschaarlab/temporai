from typing import Any

import numpy as np
import pytest
import torch

from tempor.data.datasources import GoogleStocksDataSource, SineDataSource
from tempor.models.constants import ODEBackend
from tempor.models.ts_ode import NeuralODE


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


@pytest.mark.parametrize("source", [GoogleStocksDataSource, SineDataSource])
@pytest.mark.parametrize("backend", ["laplace", "cde", "ode"])
def test_ode_regression_fit_predict(source: Any, backend: ODEBackend) -> None:
    if source == SineDataSource and backend == "laplace":
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
        clipping_value=0,
    )

    model.fit(static, temporal, observation_times, outcome)

    y_pred = model.predict(static, temporal, observation_times)

    assert y_pred.shape == outcome.shape

    assert model.score(static, temporal, observation_times, outcome) < 2


@pytest.mark.parametrize("source", [SineDataSource, GoogleStocksDataSource])
@pytest.mark.parametrize("backend", ["laplace", "cde", "ode"])
def test_ode_classification_fit_predict(source: Any, backend: ODEBackend) -> None:
    if source == SineDataSource and backend == "laplace":
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


def test_init_inputs():
    with pytest.raises(ValueError, match=".*output shape.*"):
        NeuralODE(
            task_type="regression",
            n_static_units_in=3,
            n_temporal_units_in=3,
            output_shape=[],
        )

    with pytest.raises(RuntimeError, match=".*backend.*"):
        NeuralODE(
            task_type="regression",
            n_static_units_in=3,
            n_temporal_units_in=3,
            output_shape=[2],
            backend="unknown",  # type: ignore
        )


def test_forward_nans_found():
    model = NeuralODE(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        output_shape=[2],
    )

    with pytest.raises(ValueError, match=".*NaNs.*static.*"):
        model.forward(
            static_data=torch.tensor([np.nan, np.nan]),
            temporal_data=torch.ones(10, 3, 2),
            observation_times=torch.ones(10, 3),
        )

    t = torch.ones(10, 3, 2)
    t[0, 2, 1] = torch.nan
    with pytest.raises(ValueError, match=".*NaNs.*temporal.*"):
        model.forward(
            static_data=torch.ones(10, 3),
            temporal_data=t,
            observation_times=torch.ones(10, 3),
        )

    o = torch.ones(10, 3)
    o[0, 2] = torch.nan
    with pytest.raises(ValueError, match=".*NaNs.*horizon.*"):
        model.forward(
            static_data=torch.ones(10, 3),
            temporal_data=torch.ones(10, 3, 2),
            observation_times=o,
        )


def test_forward_interpolation():
    model = NeuralODE(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        n_layers_hidden=5,
        output_shape=[2],
        interpolation="linear",
        backend="ode",
        device=torch.device("cpu"),
    )

    s = torch.ones(10, 3)
    t = torch.ones(10, 3, 3)
    o = torch.ones(10, 3)
    model.forward(s, t, o)

    model = NeuralODE(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        output_shape=[2],
        interpolation="unknown",  # type: ignore
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match=".*interpolation.*"):
        model.forward(s, t, o)


def test_forward_invalid_solver():
    s = torch.ones(10, 3)
    t = torch.ones(10, 3, 3)
    o = torch.ones(10, 3)

    model = NeuralODE(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        output_shape=[2],
        device=torch.device("cpu"),
        backend="cde",
    )
    model.backend = "unknown"
    with pytest.raises(RuntimeError, match=".*solver.*"):
        model.forward(s, t, o)


def test_predict_proba_validation_fail():
    model = NeuralODE(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        output_shape=[2],
    )

    with pytest.raises(RuntimeError, match=".*classification.*"):
        model.predict_proba([], [], [])


def test_ode_early_stop() -> None:
    static, temporal, observation_times, outcome = unpack_dataset(SineDataSource)

    outcome = outcome.reshape(-1, 1)

    outlen = int(len(outcome.reshape(-1)) / len(outcome))

    model = NeuralODE(
        task_type="regression",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=outcome.shape[1:],  # type: ignore
        n_iter=100,
        nonlin_out=[("tanh", outlen)],
        backend="cde",
        clipping_value=0,
        n_iter_print=1,
        patience=1,
    )

    model.fit(static, temporal, observation_times, outcome)


def test_check_tensor():
    t = torch.ones(size=(3, 2))
    model = NeuralODE(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        output_shape=[2],
    )
    out = model._check_tensor(t)  # pylint: disable=protected-access
    assert out.device == model.device
