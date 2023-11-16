# TODO

# import numpy as np
# import pytest

# from tempor import plugin_loader

# TEST_CASE_PAIRS = [
#     # Cases: class labels.
#     (  # 1. Perfect prediction:
#         np.array([2.0, 7.5, 2.1, 9.3]),
#         np.array([2.0, 7.5, 2.1, 9.3]),
#     ),
#     (  # 3. Non-perfect prediction:
#         np.array([2.0, 7.5, 2.1, 9.3]),
#         np.array([-1.2, 7.1, 2.8, 11.2]),
#     ),
#     # NOTE: Any additional cases here.
# ]

# EXPECTED_VALUES = {
#     "c_index": [0.0, 3.6249],
#     "brier_score": [0.0, 1.5499],
#     # NOTE: Any additional metrics here.
#     # "<metric_name>": [...expected values corresponding to the test cases above...]
# }

# METRIC_NAMES = list(EXPECTED_VALUES.keys())


# @pytest.mark.parametrize("metric_name", METRIC_NAMES)
# @pytest.mark.parametrize("case_idx, actual, predicted", [(idx, a, p) for idx, (a, p) in enumerate(TEST_CASE_PAIRS)])
# def test_accuracy_metric(metric_name, case_idx, actual, predicted):
#     metric = plugin_loader.get(f"time_to_event.{metric_name}", plugin_type="metric")
#     result = metric.evaluate(actual, predicted)

#     expected = EXPECTED_VALUES[metric_name][case_idx]

#     assert metric.direction in ("maximize", "minimize")
#     np.testing.assert_almost_equal(result, expected, 4)
