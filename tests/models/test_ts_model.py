from typing import Any

import numpy as np
import pytest
from typing_extensions import get_args

from tempor.models.ts_model import ModelTaskType, TimeSeriesModel, TSModelMode
from tempor.utils.dataloaders.google_stocks import GoogleStocksDataLoader
from tempor.utils.dataloaders.sine import SineDataLoader


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
    )

    static_data = np.concatenate([static, static_fake])
    temporal_data = np.concatenate([temporal, temporal_fake])
    observation_times = np.concatenate([observation_times, observation_times])

    model.fit(static_data, temporal_data, observation_times, y)

    y_pred = model.predict(static_data, temporal_data, observation_times)

    assert y_pred.shape == y.shape

    assert model.score(static_data, temporal_data, observation_times, y) <= 1
