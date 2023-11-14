# pylint: disable=protected-access

import numpy as np
import pytest

from tempor import plugin_loader
from tempor.metrics.prediction.one_off import plugin_classification

# ------------------------------------------------------------------------------
# Test utilities.


class TestUtilities:
    class TestCastToYPred:
        def test_with_probabilities(self):
            y_pred_proba = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
            expected = np.array([1, 0, 1])
            result = plugin_classification._cast_to_y_pred(y_pred_proba)
            np.testing.assert_array_equal(result, expected)

        def test_with_class_labels(self):
            y_pred_proba = np.array([1, 0, 1])
            expected = np.array([1, 0, 1])
            result = plugin_classification._cast_to_y_pred(y_pred_proba)
            np.testing.assert_array_equal(result, expected)

        def test_with_invalid_dimensions(self):
            y_pred_proba = np.array([[[0.1, 0.9], [0.8, 0.2]]])
            with pytest.raises(ValueError):
                plugin_classification._cast_to_y_pred(y_pred_proba)

    def test_prep_auc_multiclass_fails(self):
        with pytest.raises(ValueError, match=".*NaN.*"):
            plugin_classification._prep_auc_multiclass(np.array([1, 0]), np.array([[0.1, 0.9], [np.nan, 0.2]]))

    class TestGetYPredProbaHlpr:
        def test_2class_2prob(self):
            in_ = np.array([[0.1, 0.9], [0.3, 0.7], [0.11, 0.89]])
            exp = np.array([[0.9], [0.7], [0.89]]).ravel()
            out = plugin_classification._get_y_pred_proba_hlpr(in_, nclasses=2)
            assert (out == exp).all()

        def test_2class_1prob(self):
            in_ = np.array([[0.1], [0.3], [0.11]])
            out = plugin_classification._get_y_pred_proba_hlpr(in_, nclasses=2)
            assert (out == in_).all()


# ------------------------------------------------------------------------------
# Test the actual metrics.


TEST_CASE_PAIRS = [
    # Cases: class labels.
    (  # 1. Perfect prediction:
        np.array([1, 0, 1, 0]),
        np.array([1, 0, 1, 0]),
    ),
    (  # 2. Completely incorrect prediction:
        np.array([1, 0, 1, 0]),
        np.array([0, 1, 0, 1]),
    ),
    (  # 3. Partially correct prediction:
        np.array([1, 0, 1, 0]),
        np.array([1, 1, 0, 0]),
    ),
    # Cases: class probabilities.
    (  # 4. Perfect prediction:
        np.array([1, 0, 1, 0]),
        np.array([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]),
    ),
    (  # 5. Completely incorrect prediction:
        np.array([1, 0, 1, 0]),
        np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.4, 0.6]]),
    ),
    (  # 6. Partially correct prediction:
        np.array([1, 0, 1, 0]),
        np.array([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]]),
    ),
    # Cases: 2+ classes.
    (  # 7. Perfect prediction:
        np.array([1, 0, 2, 0]),
        np.array([[0.1, 0.9, 0.0], [0.7, 0.3, 0.0], [0.0, 0.0, 1.0], [0.6, 0.4, 0.0]]),
    ),
    (  # 8. 1 of 4 is correct prediction:
        np.array([1, 0, 2, 0]),
        np.array([[0.9, 0.1, 0.0], [0.3, 0.7, 0.0], [0.0, 0.0, 1.0], [0.4, 0.6, 0.0]]),
    ),
    # NOTE: Any additional cases here.
]

EXPECTED_VALUES = {
    "accuracy": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.25],
    "f1_score_micro": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.25],
    "f1_score_macro": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.3333],
    "f1_score_weighted": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.25],
    "kappa": [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -0.0909],
    "kappa_quadratic": [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.4],
    "recall_micro": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.25],
    "recall_macro": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.3333],
    "recall_weighted": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.25],
    "precision_micro": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.25],
    "precision_macro": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.3333],
    "precision_weighted": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.25],
    "mcc": [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -0.1],
    "aucprc": [1.0, 0.5, 0.5, 1.0, 0.41666, 0.75, 1.0, 0.61785],
    "aucroc": [1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.71875],
    # NOTE: Any additional metrics here.
    # "<metric_name>": [...expected values corresponding to the test cases above...]
}

METRIC_NAMES = list(EXPECTED_VALUES.keys())


@pytest.mark.parametrize("metric_name", METRIC_NAMES)
@pytest.mark.parametrize("case_idx, actual, predicted", [(idx, a, p) for idx, (a, p) in enumerate(TEST_CASE_PAIRS)])
def test_accuracy_metric(metric_name, case_idx, actual, predicted):
    metric = plugin_loader.get(f"prediction.one_off.classification.{metric_name}", plugin_type="metric")
    result = metric.evaluate(actual, predicted)

    expected = EXPECTED_VALUES[metric_name][case_idx]

    assert metric.direction in ("maximize", "minimize")
    np.testing.assert_almost_equal(result, expected, 4)
