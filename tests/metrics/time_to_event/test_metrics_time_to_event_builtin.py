from typing import NamedTuple, Optional

import numpy as np
import pytest

from tempor import plugin_loader
from tempor.data import data_typing
from tempor.metrics import metric_typing


class TestData(NamedTuple):
    actual: metric_typing.EventArrayTimeArray
    predicted: np.ndarray
    horizons: data_typing.TimeIndex
    actual_train: Optional[metric_typing.EventArrayTimeArray] = None


TEST_CASE_PAIRS = [
    TestData(
        actual=(
            np.asarray([1, 0, 1, 1, 1, 0]),
            np.asarray([100.0, 110.0, 108.0, 99.0, 101.0, 140.0]),
        ),
        # predicted: (n_samples, n_horizons_timesteps, n_features)
        predicted=np.expand_dims(
            np.asarray(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                ]
            ).T,
            axis=-1,
        ),
        horizons=[100.0, 115.0, 130.0],
        actual_train=(
            np.asarray([1, 0, 1, 0, 0, 0, 1, 1, 0, 1]),
            np.asarray([100.0, 102.0, 107.0, 120.0, 103.0, 133.0, 114.0, 170.0, 113.0, 124.0]),
        ),
    ),
    # NOTE: Any additional cases here.
]

EXPECTED_VALUES = {
    "c_index": [
        [0.6, 0.4346, 0.4346],
    ],
    "brier_score": [
        [0.285, 0.1657, 0.2239],
    ],
    # NOTE: Any additional metrics here.
    # "<metric_name>": [...expected arrays (metric value per horizon) corresponding to the test cases above...]
}

METRIC_NAMES = list(EXPECTED_VALUES.keys())


@pytest.mark.parametrize("metric_name", METRIC_NAMES)
@pytest.mark.parametrize("case_idx, case_data", enumerate(TEST_CASE_PAIRS))
def test_accuracy_metric(metric_name, case_idx, case_data):
    metric = plugin_loader.get(f"time_to_event.{metric_name}", plugin_type="metric")

    actual, predicted, horizons, actual_train = case_data
    result = metric.evaluate(actual, predicted, horizons, actual_train=actual_train)

    expected = EXPECTED_VALUES[metric_name][case_idx]

    assert metric.direction in ("maximize", "minimize")
    np.testing.assert_almost_equal(result, expected, 4)
