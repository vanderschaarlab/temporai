"""Any utilities for the ``benchmark`` package directory."""

from typing import Tuple, cast

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from tempor.log import logger as log


def get_y_pred_proba_hlpr(y_pred_proba: np.ndarray, nclasses: int) -> np.ndarray:
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


def evaluate_auc_multiclass(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Tuple[float, float]:
    """Helper for evaluating AUCROC/AUCPRC for any number of classes."""

    y_test = np.asarray(y_test)
    y_pred_proba = np.asarray(y_pred_proba)

    nnan = sum(np.ravel(np.isnan(y_pred_proba)))

    if nnan:
        raise ValueError("nan in predictions. aborting")

    n_classes = len(set(np.ravel(y_test)))
    classes = sorted(set(np.ravel(y_test)))

    y_pred_proba_tmp = get_y_pred_proba_hlpr(y_pred_proba, n_classes)

    if n_classes > 2:
        log.debug(f"+evaluate_auc {y_test.shape} {y_pred_proba_tmp.shape}")

        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()
        roc_auc: dict = dict()

        y_test = cast(np.ndarray, label_binarize(y_test, classes=classes, sparse_output=False))

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_proba_tmp.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred_proba_tmp.ravel())

        average_precision["micro"] = average_precision_score(y_test, y_pred_proba_tmp, average="micro")

        aucroc = roc_auc["micro"]
        aucprc = average_precision["micro"]
    else:
        aucroc = roc_auc_score(np.ravel(y_test), y_pred_proba_tmp, multi_class="ovr")
        aucprc = average_precision_score(np.ravel(y_test), y_pred_proba_tmp)

    return float(aucroc), float(aucprc)


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    """Return score as mean and confidence interval using the 1.96 rule: ``(mean, 1.96 * std / sqrt(n))``.
    See e.g. https://math.stackexchange.com/a/1572814.

    Args:
        metric (np.ndarray): Input metric.

    Returns:
        Tuple[float, float]: The score as ``(mean, confidence interval)``.
    """
    percentile_val = 1.96
    return (float(np.mean(metric)), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    """Print score as ``mean +/- range`` (3 decimal places).

    Args:
        score (Tuple[float, float]): The score to print as ``(mean, range)``.

    Returns:
        str: The formatted string.
    """
    return str(round(score[0], 3)) + " +/- " + str(round(score[1], 3))
