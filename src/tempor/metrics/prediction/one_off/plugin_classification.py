"""Module with built-in metric plugins for the category: prediction -> one-off -> classification."""

from typing import Any, List, Tuple, cast

import numpy as np
import sklearn.metrics
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from tempor.core import plugins
from tempor.metrics import metric


def _cast_to_y_pred(y_pred_proba: np.ndarray) -> np.ndarray:
    """Turn an array of class probabilities ``y_pred_proba`` into an array of class predictions ``y_pred``.

    Args:
        y_pred_proba (np.ndarray): Predicted probabilities.

    Returns:
        np.ndarray: Predicted classes.
    """
    if y_pred_proba.ndim == 2:
        # Interpret as probabilities.
        y_pred = np.argmax(np.asarray(y_pred_proba), axis=1)
    elif y_pred_proba.ndim == 1:
        # Interpret as class labels.
        y_pred = np.asarray(y_pred_proba)
    else:
        raise ValueError(f"Invalid shape of y_pred_proba: {y_pred_proba.shape}")
    return y_pred


def _get_y_pred_proba_hlpr(y_pred_proba: np.ndarray, nclasses: int) -> np.ndarray:
    """A helper utility for inferring the correct y_pred_proba for multiclass situations, specifically in the case
    of binary classification. See source code for the specifics.

    Args:
        y_pred_proba (np.ndarray): Predicted probabilities.
        nclasses (int): Number of classes.

    Returns:
        np.ndarray: The correctly inferred ``y_pred_proba``.
    """
    if nclasses == 2:
        if len(y_pred_proba.shape) < 2:
            return y_pred_proba

        if y_pred_proba.shape[1] == 2:
            return y_pred_proba[:, 1]

    return y_pred_proba


def _prep_auc_multiclass(y_test: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, List]:
    y_test = np.asarray(y_test)
    y_pred_proba = np.asarray(y_pred_proba)

    nnan = sum(np.ravel(np.isnan(y_pred_proba)))
    if nnan:
        raise ValueError("NaNs in predictions, aborting.")

    n_classes = len(set(np.ravel(y_test)))
    classes = sorted(set(np.ravel(y_test)))

    y_pred_proba_tmp = _get_y_pred_proba_hlpr(y_pred_proba, n_classes)

    return y_test, y_pred_proba_tmp, n_classes, classes


def _evaluate_aucroc_multiclass(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Helper for evaluating AUCROC for any number of classes."""

    y_test, y_pred_proba_tmp, n_classes, classes = _prep_auc_multiclass(y_test, y_pred_proba)

    if n_classes > 2:
        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        roc_auc: dict = dict()

        y_test = cast(np.ndarray, label_binarize(y_test, classes=classes, sparse_output=False))

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_proba_tmp.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred_proba_tmp.ravel())

        aucroc = roc_auc["micro"]
    else:
        aucroc = roc_auc_score(np.ravel(y_test), y_pred_proba_tmp, multi_class="ovr")

    return float(aucroc)


def _evaluate_aucprc_multiclass(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Helper for evaluating AUCPRC for any number of classes."""

    y_test, y_pred_proba_tmp, n_classes, classes = _prep_auc_multiclass(y_test, y_pred_proba)

    if n_classes > 2:
        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()

        y_test = cast(np.ndarray, label_binarize(y_test, classes=classes, sparse_output=False))

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_proba_tmp.ravel())
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred_proba_tmp.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_pred_proba_tmp, average="micro")

        aucprc = average_precision["micro"]
    else:
        aucprc = average_precision_score(np.ravel(y_test), y_pred_proba_tmp)

    return float(aucprc)


@plugins.register_plugin(name="accuracy", category="prediction.one_off.classification", plugin_type="metric")
class AccuracyMetric(metric.OneOffClassificationMetric):
    """Accuracy classification score."""

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.accuracy_score(actual, _cast_to_y_pred(predicted)),
        )


@plugins.register_plugin(name="f1_score_micro", category="prediction.one_off.classification", plugin_type="metric")
class F1ScoreMicroMetric(metric.OneOffClassificationMetric):
    """F1 score is a harmonic mean of the precision and recall. This version uses the ``"micro"`` average: calculate
    metrics globally by counting the total true positives, false negatives and false positives.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.f1_score(
                actual,
                _cast_to_y_pred(predicted),
                average="micro",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="f1_score_macro", category="prediction.one_off.classification", plugin_type="metric")
class F1ScoreMacroMetric(metric.OneOffClassificationMetric):
    """F1 score is a harmonic mean of the precision and recall. This version uses the ``"macro"`` average: calculate
    metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.f1_score(
                actual,
                _cast_to_y_pred(predicted),
                average="macro",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="f1_score_weighted", category="prediction.one_off.classification", plugin_type="metric")
class F1ScoreWeightedMetric(metric.OneOffClassificationMetric):
    """F1 score is a harmonic mean of the precision and recall. This version uses the ``"weighted"`` average: calculate
    metrics for each label, and find their average weighted by support (the number of true instances for each label).
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.f1_score(
                actual,
                _cast_to_y_pred(predicted),
                average="weighted",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="kappa", category="prediction.one_off.classification", plugin_type="metric")
class KappaMetric(metric.OneOffClassificationMetric):
    """Computes Cohen's kappa, a score that expresses the level of agreement between two annotators on a classification
    problem.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.cohen_kappa_score(actual, _cast_to_y_pred(predicted)),
        )


@plugins.register_plugin(name="kappa_quadratic", category="prediction.one_off.classification", plugin_type="metric")
class KappaQuadraticMetric(metric.OneOffClassificationMetric):
    """Computes Cohen's kappa, a score that expresses the level of agreement between two annotators on a classification
    problem. Weighted using the `"quadratic"` weighting.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.cohen_kappa_score(actual, _cast_to_y_pred(predicted), weights="quadratic"),
        )


@plugins.register_plugin(name="recall_micro", category="prediction.one_off.classification", plugin_type="metric")
class RecallMicroMetric(metric.OneOffClassificationMetric):
    """Recall is defined as the number of true positives over the number of true positives plus the number of false
    negatives. This version (micro) calculates metrics globally by counting the total true positives.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.recall_score(
                actual,
                _cast_to_y_pred(predicted),
                average="micro",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="recall_macro", category="prediction.one_off.classification", plugin_type="metric")
class RecallMacroMetric(metric.OneOffClassificationMetric):
    """Recall is defined as the number of true positives over the number of true positives plus the number of false
    negatives. This version (macro) calculates metrics for each label, and finds their unweighted mean.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.recall_score(
                actual,
                _cast_to_y_pred(predicted),
                average="macro",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="recall_weighted", category="prediction.one_off.classification", plugin_type="metric")
class RecallWeightedMetric(metric.OneOffClassificationMetric):
    """Recall is defined as the number of true positives over the number of true positives plus the number of false
    negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.recall_score(
                actual,
                _cast_to_y_pred(predicted),
                average="weighted",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="precision_micro", category="prediction.one_off.classification", plugin_type="metric")
class PrecisionMicroMetric(metric.OneOffClassificationMetric):
    """Precision is defined as the number of true positives over the number of true positives plus the number of false
    positives. This version (micro) calculates metrics globally by counting the total true positives.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.recall_score(
                actual,
                _cast_to_y_pred(predicted),
                average="micro",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="precision_macro", category="prediction.one_off.classification", plugin_type="metric")
class PrecisionMacroMetric(metric.OneOffClassificationMetric):
    """Precision is defined as the number of true positives over the number of true positives plus the number of
    false positives. This version (macro) calculates metrics for each label, and finds their unweighted mean.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.recall_score(
                actual,
                _cast_to_y_pred(predicted),
                average="macro",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="precision_weighted", category="prediction.one_off.classification", plugin_type="metric")
class PrecisionWeightedMetric(metric.OneOffClassificationMetric):
    """Precision is defined as the number of true positives over the number of true positives plus the number of
    false positives. This version (weighted) calculates metrics for each label, and find their average weighted
    by support.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.recall_score(
                actual,
                _cast_to_y_pred(predicted),
                average="weighted",
                zero_division=0,
            ),
        )


@plugins.register_plugin(name="mcc", category="prediction.one_off.classification", plugin_type="metric")
class MccMetric(metric.OneOffClassificationMetric):
    """The Matthews Correlation Coefficient is used in machine learning as a measure of the quality of binary and
    multiclass classifications. It takes into account true and false positives and negatives and is generally
    regarded as a balanced measure which can be used even if the classes are of very different sizes.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return cast(
            float,
            sklearn.metrics.matthews_corrcoef(actual, _cast_to_y_pred(predicted)),
        )


@plugins.register_plugin(name="aucprc", category="prediction.one_off.classification", plugin_type="metric")
class AucPrcMetric(metric.OneOffClassificationMetric):
    """The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each
    threshold, with the increase in recall from the previous threshold used as the weight.
    """

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return _evaluate_aucprc_multiclass(actual, predicted)


@plugins.register_plugin(name="aucroc", category="prediction.one_off.classification", plugin_type="metric")
class AucRocMetric(metric.OneOffClassificationMetric):
    """The Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores."""

    @property
    def direction(self) -> metric.MetricDirection:  # noqa: D102
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> float:
        return _evaluate_aucroc_multiclass(actual, predicted)
