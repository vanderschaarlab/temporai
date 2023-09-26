# pylint: disable=redefined-outer-name

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from tempor.models.ddh import DynamicDeepHitLayers, DynamicDeepHitModel
from tempor.utils.dataloaders import PBCDataLoader


@pytest.fixture(scope="module")
def get_test_data():
    # This is module-scoped such that tests can run quicker.

    # Some change here.

    data = PBCDataLoader().load()
    assert data.predictive.targets is not None

    x: Any = [df.to_numpy() for df in data.time_series.list_of_dataframes()]
    x = np.array(x, dtype=object)
    t, e = (df.to_numpy().reshape((-1,)) for df in data.predictive.targets.split_as_two_dataframes())

    event0_times = data.predictive.targets.split_as_two_dataframes()[0].to_numpy().reshape((-1,))
    horizons = np.quantile(event0_times, [0.25, 0.5, 0.75]).tolist()

    return x, t, e, horizons


# Test DynamicDeepHitModel:


@pytest.mark.parametrize(
    "rnn_mode",
    [
        "GRU",
        "LSTM",
        "RNN",
        "Transformer",
    ],
)
@pytest.mark.parametrize(
    "output_mode",
    [
        "MLP",
    ],
)
def test_ddh_predict_rnn_modes(
    get_test_data,
    rnn_mode,
    output_mode,
) -> None:
    x, t, e, horizons = get_test_data

    model = DynamicDeepHitModel(n_iter=10, rnn_mode=rnn_mode, output_mode=output_mode)
    output = model.fit(x=x, t=t, e=e).predict_risk(x=x, t=horizons)

    assert output.shape == (len(x), len(horizons))


@pytest.mark.parametrize(
    "rnn_mode",
    [
        "GRU",
    ],
)
@pytest.mark.parametrize(
    "output_mode",
    [
        "MLP",
        "LSTM",
        "GRU",
        "RNN",
        "Transformer",
        "TCN",
        "InceptionTime",
        "InceptionTimePlus",
        "ResCNN",
        "XCM",
    ],
)
def test_ddh_predict_output_modes(
    get_test_data,
    rnn_mode,
    output_mode,
) -> None:
    x, t, e, horizons = get_test_data

    model = DynamicDeepHitModel(n_iter=10, rnn_mode=rnn_mode, output_mode=output_mode)
    output = model.fit(x=x, t=t, e=e).predict_risk(x=x, t=horizons)

    assert output.shape == (len(x), len(horizons))


def test_val_set_fail(get_test_data):
    x, t, e, _ = get_test_data
    model = DynamicDeepHitModel(n_iter=10, clipping_value=0, val_size=0.0001)
    with pytest.raises(RuntimeError, match=".*[Vv]alidation.*"):
        model.fit(x=x, t=t, e=e)


def test_val_set_too_small_warning(get_test_data):
    x, t, e, _ = get_test_data
    model = DynamicDeepHitModel(n_iter=10, clipping_value=0, val_size=(5 / len(x)))
    with pytest.warns(RuntimeWarning, match=".*[Vv]alidation.*small.*"):
        model.fit(x=x, t=t, e=e)


def test_ddh_predict_more_cases(get_test_data):
    x, t, e, horizons = get_test_data

    # No clipping.
    model = DynamicDeepHitModel(n_iter=10, clipping_value=0)
    output = model.fit(x=x, t=t, e=e).predict_risk(x=x, t=horizons)
    assert output.shape == (len(x), len(horizons))

    # Force early stop.
    model = DynamicDeepHitModel(n_iter=10, patience=0)
    output = model.fit(x=x, t=t, e=e).predict_risk(x=x, t=horizons)
    assert output.shape == (len(x), len(horizons))


def test_ddh_not_fit(get_test_data):
    x, t, *_ = get_test_data

    model = DynamicDeepHitModel(n_iter=10, clipping_value=0)
    with pytest.raises(RuntimeError, match=".*fit.*"):
        model.predict_emb(x)
    with pytest.raises(RuntimeError, match=".*fit.*"):
        model.predict_survival(x, t)


def test_loss_fails(get_test_data):
    import torch

    x, t, e, _ = get_test_data

    model = DynamicDeepHitModel(n_iter=10, clipping_value=0)

    with pytest.raises(RuntimeError, match=".*loss.*"):
        model.total_loss(x, t, e)

    model.model = Mock(return_value=torch.tensor([np.nan, np.nan]))
    x = torch.ones(size=[10, 2])

    with pytest.raises(RuntimeError, match=".*NaNs.*"):
        model.total_loss(x, t, e)


def test_ranking_loss_edge_case():
    # if torch.sum(t > ti) > 0 : False
    import torch

    model = DynamicDeepHitModel(n_iter=10, clipping_value=0)

    model.ranking_loss(
        cif=[
            torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ],
        t=torch.tensor([2, 1, 1]),
        e=torch.tensor([1, 0, 1]),
    )


def test_ddh_predict_survival_all_step(get_test_data):
    x, t, e, horizons = get_test_data

    model = DynamicDeepHitModel(n_iter=10, clipping_value=0)
    output = model.fit(x=x, t=t, e=e).predict_survival(x=x, t=horizons, all_step=True)
    assert output.shape[1] == len(horizons)
    assert output.shape[0] >= len(x)


# Test DynamicDeepHitLayers:


def test_ddh_layers_rnn_type():
    with pytest.raises(RuntimeError, match=".*rnn.*type.*"):
        DynamicDeepHitLayers(input_dim=10, seq_len=10, output_dim=10, layers_rnn=1, hidden_rnn=2, rnn_type="unknown")
