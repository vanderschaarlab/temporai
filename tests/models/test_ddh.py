# pylint: disable=redefined-outer-name

from typing import Any

import numpy as np
import pytest

from tempor.models.ddh import DynamicDeepHitModel
from tempor.utils.dataloaders import PBCDataLoader


@pytest.fixture(scope="module")
def get_test_data():
    # This is module-scoped such that tests can run quicker.

    data = PBCDataLoader().load()
    x: Any = [df.to_numpy() for df in data.time_series.list_of_dataframes()]
    x = np.array(x, dtype=object)
    t, e = (df.to_numpy().reshape((-1,)) for df in data.predictive.targets.split_as_two_dataframes())

    event0_times = data.predictive.targets.split_as_two_dataframes()[0].to_numpy().reshape((-1,))
    horizons = np.quantile(event0_times, [0.25, 0.5, 0.75]).tolist()

    return x, t, e, horizons


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
