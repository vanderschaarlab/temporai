from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from typing_extensions import get_args

from tempor.models.ts_model import ModelTaskType, TimeSeriesLayer, TimeSeriesModel, TSModelMode, WindowLinearLayer
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


@pytest.mark.parametrize("mode", get_args(TSModelMode))
@pytest.mark.parametrize("task_type", get_args(ModelTaskType))
def test_rnn_sanity(mode: TSModelMode, task_type: ModelTaskType) -> None:
    model = TimeSeriesModel(
        task_type=task_type,
        n_static_units_in=3,
        n_temporal_units_in=4,
        n_temporal_window=2,
        output_shape=[2],
        n_iter=11,
        n_static_units_hidden=41,
        n_temporal_units_hidden=42,
        n_static_layers_hidden=2,
        n_temporal_layers_hidden=3,
        mode=mode,
        n_iter_print=12,
        batch_size=123,
        lr=1e-2,
        weight_decay=1e-2,
    )

    assert model.n_iter == 11
    assert model.n_static_units_in == 3
    assert model.n_temporal_units_in == 4
    assert model.n_units_out == 2
    assert model.output_shape == [2]
    assert model.n_static_units_hidden == 41
    assert model.n_temporal_units_hidden == 42
    assert model.n_static_layers_hidden == 2
    assert model.n_temporal_layers_hidden == 3
    assert model.mode == mode
    assert model.n_iter_print == 12
    assert model.batch_size == 123


@pytest.mark.parametrize("mode", get_args(TSModelMode))
@pytest.mark.parametrize("source", [GoogleStocksDataLoader, SineDataLoader])
@pytest.mark.parametrize("use_horizon_condition", [True, False])
def test_rnn_regression_fit_predict(mode: TSModelMode, source: Any, use_horizon_condition: bool) -> None:
    static, temporal, observation_times, outcome = unpack_dataset(source)

    outcome = outcome.reshape(-1, 1)

    outlen = int(len(outcome.reshape(-1)) / len(outcome))

    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        n_temporal_window=temporal.shape[1],
        output_shape=outcome.shape[1:],
        n_iter=10,
        nonlin_out=[("tanh", outlen)],
        mode=mode,
        use_horizon_condition=use_horizon_condition,
    )

    model.fit(static, temporal, observation_times, outcome)

    y_pred = model.predict(static, temporal, observation_times)

    assert y_pred.shape == outcome.shape

    assert model.score(static, temporal, observation_times, outcome) < 2


@pytest.mark.parametrize("mode", get_args(TSModelMode))
@pytest.mark.parametrize("source", [SineDataLoader, GoogleStocksDataLoader])
def test_rnn_classification_fit_predict(mode: TSModelMode, source: Any) -> None:
    static, temporal, observation_times, outcome = unpack_dataset(source)  # pylint: disable=unused-variable
    static_fake, temporal_fake = np.random.randn(*static.shape), np.random.randn(*temporal.shape)

    y = np.asarray([1] * len(static) + [0] * len(static_fake))  # type: ignore

    model = TimeSeriesModel(
        task_type="classification",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        n_temporal_window=temporal.shape[1],
        output_shape=[2],
        n_iter=10,
        mode=mode,
        clipping_value=0,
    )

    static_data = np.concatenate([static, static_fake])
    temporal_data = np.concatenate([temporal, temporal_fake])
    observation_times = np.concatenate([observation_times, observation_times])

    model.fit(static_data, temporal_data, observation_times, y)

    y_pred = model.predict(static_data, temporal_data, observation_times)

    assert y_pred.shape == y.shape

    assert model.score(static_data, temporal_data, observation_times, y) <= 1


def test_init_inputs():
    with pytest.raises(ValueError, match=".*shape.*"):
        TimeSeriesModel(
            task_type="regression",
            n_static_units_in=3,
            n_temporal_units_in=3,
            n_temporal_window=10,
            output_shape=[],
        )

    loss = Mock()
    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        n_temporal_window=10,
        output_shape=[2],
        loss=loss,
    )
    assert model.loss == loss

    with pytest.raises(RuntimeError, match=".*mismatch.*"):
        TimeSeriesModel(
            task_type="regression",
            n_static_units_in=3,
            n_temporal_units_in=3,
            n_temporal_window=10,
            output_shape=[1, 7],
            loss=loss,
            nonlin_out=[("tanh", 1), ("tanh", 2)],
        )


def test_forward_nans_found():
    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        n_temporal_window=2,
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


def test_predict_proba_validation_fail():
    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        n_temporal_window=2,
        output_shape=[2],
    )

    with pytest.raises(RuntimeError, match=".*classification.*"):
        model.predict_proba([], [], [])


def test_check_tensor():
    t = torch.ones(size=(3, 2))
    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=3,
        n_temporal_units_in=3,
        n_temporal_window=2,
        output_shape=[2],
    )
    out = model._check_tensor(t)  # pylint: disable=protected-access
    assert out.device == model.device


def test_time_series_layer_unknown_mode():
    with pytest.raises(RuntimeError, match=".*mode.*"):
        TimeSeriesLayer(
            n_static_units_in=3,
            n_temporal_units_in=3,
            n_temporal_window=2,
            n_units_out=2,
            mode="unknown",  # type: ignore
        )


@pytest.mark.parametrize("mode", ["RNN", "InceptionTime"])
def test_time_series_layer_forward_nans(mode):
    tsl = TimeSeriesLayer(
        n_static_units_in=0,
        n_temporal_units_in=3,
        n_temporal_window=2,
        n_units_out=2,
        mode=mode,
        device=torch.device("cpu"),
    )
    s = torch.ones(size=(0,))
    t = torch.ones(size=(10, 2, 3))
    t[0, 0, 0] = torch.nan
    with pytest.raises(RuntimeError, match=".*NaNs.*"):
        tsl.forward(s, t)


def test_windowed_layer_forward_nans():
    tsl = WindowLinearLayer(
        n_static_units_in=2,
        n_temporal_units_in=3,
        window_size=2,
        n_units_out=2,
        device=torch.device("cpu"),
    )
    s = torch.ones(size=(5, 2))
    t = torch.ones(size=(10, 2, 3))
    t[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match=".*mismatch.*"):
        tsl.forward(s, t)
