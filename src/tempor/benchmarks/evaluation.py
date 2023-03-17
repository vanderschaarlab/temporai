# stdlib
import copy
from typing import Any, Dict, Union

# third party
import numpy as np
from pydantic import validate_arguments
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from tempor.data import dataset
from tempor.log import logger as log
from tempor.models.utils import enable_reproducibility

from .utils import evaluate_auc_multiclass, generate_score, print_score

classifier_supported_metrics = [
    "aucroc",
    "aucprc",
    "accuracy",
    "f1_score_micro",
    "f1_score_macro",
    "f1_score_weighted",
    "kappa",
    "kappa_quadratic",
    "precision_micro",
    "precision_macro",
    "precision_weighted",
    "recall_micro",
    "recall_macro",
    "recall_weighted",
    "mcc",
]
regression_supported_metrics = ["mse", "mae", "r2"]


class classifier_metrics:
    """Helper class for evaluating the performance of the classifier.

    Args:
        metric: list, default=["aucroc", "aucprc", "accuracy", "f1_score_micro", "f1_score_macro", "f1_score_weighted",  "kappa", "precision_micro", "precision_macro", "precision_weighted", "recall_micro", "recall_macro", "recall_weighted",  "mcc",]
            The type of metric to use for evaluation.
            Potential values:
                - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
                - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
                - "accuracy" : Accuracy classification score.
                - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
                - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                - "kappa", "kappa_quadratic":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
                - "precision_micro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(micro) calculates metrics globally by counting the total true positives.
                - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
                - "precision_weighted": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
                - "recall_micro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(micro) calculates metrics globally by counting the total true positives.
                - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
                - "recall_weighted": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
                - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
    """

    def __init__(self, metric: Union[str, list] = classifier_supported_metrics) -> None:
        if isinstance(metric, str):
            self.metrics = [metric]
        else:
            self.metrics = metric

    def get_metric(self) -> Union[str, list]:
        return self.metrics

    def score_proba(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        if y_test is None or y_pred_proba is None:
            raise RuntimeError("Invalid input for score_proba")

        results = {}
        y_pred = np.argmax(np.asarray(y_pred_proba), axis=1)
        for metric in self.metrics:
            if metric == "aucprc":
                results[metric] = self.average_precision_score(y_test, y_pred_proba)
            elif metric == "aucroc":
                results[metric] = self.roc_auc_score(y_test, y_pred_proba)
            elif metric == "accuracy":
                results[metric] = accuracy_score(y_test, y_pred)
            elif metric == "f1_score_micro":
                results[metric] = f1_score(y_test, y_pred, average="micro", zero_division=0)
            elif metric == "f1_score_macro":
                results[metric] = f1_score(y_test, y_pred, average="macro", zero_division=0)
            elif metric == "f1_score_weighted":
                results[metric] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            elif metric == "kappa":
                results[metric] = cohen_kappa_score(y_test, y_pred)
            elif metric == "kappa_quadratic":
                results[metric] = cohen_kappa_score(y_test, y_pred, weights="quadratic")
            elif metric == "recall_micro":
                results[metric] = recall_score(y_test, y_pred, average="micro", zero_division=0)
            elif metric == "recall_macro":
                results[metric] = recall_score(y_test, y_pred, average="macro", zero_division=0)
            elif metric == "recall_weighted":
                results[metric] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            elif metric == "precision_micro":
                results[metric] = precision_score(y_test, y_pred, average="micro", zero_division=0)
            elif metric == "precision_macro":
                results[metric] = precision_score(y_test, y_pred, average="macro", zero_division=0)
            elif metric == "precision_weighted":
                results[metric] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            elif metric == "mcc":
                results[metric] = matthews_corrcoef(y_test, y_pred)
            else:
                raise ValueError(f"invalid metric {metric}")

        log.debug(f"evaluate_classifier: {results}")
        return results

    def roc_auc_score(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return evaluate_auc_multiclass(y_test, y_pred_proba)[0]

    def average_precision_score(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> float:

        return evaluate_auc_multiclass(y_test, y_pred_proba)[1]


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_classifier(
    estimator: Any,
    data: dataset.Dataset,
    n_splits: int = 3,
    seed: int = 0,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    """Helper for evaluating classifiers.

    Args:
        estimator:
            Baseline model to evaluate - must be unfitted.
        data: Dataset:
            The dataset
        n_splits: int
            cross-validation folds
        seed: int
            Random seed

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
        Both "raw" and "str" nodes contain the following metrics:
            - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
            - "accuracy" : Accuracy classification score.
            - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
            - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
            - "kappa":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
            - "precision_micro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(micro) calculates metrics globally by counting the total true positives.
            - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "precision_weighted": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "recall_micro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(micro) calculates metrics globally by counting the total true positives.
            - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "recall_weighted": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.

    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    enable_reproducibility(seed)

    results = {}

    evaluator = classifier_metrics()
    for metric in classifier_supported_metrics:
        results[metric] = np.zeros(n_splits)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if data.predictive is None:
        raise ValueError("No targets to use for train/test")

    labels = data.predictive.targets.numpy().squeeze()
    if len(labels.shape) > 1:
        raise ValueError("Classifier evaluation expects 1D output")

    indx = 0
    for train_data, test_data in data.split(splitter=splitter, y=labels):
        model = copy.deepcopy(estimator)
        model.fit(train_data)

        if test_data.predictive is None:
            raise ValueError("No targets to use for testing")

        test_labels = test_data.predictive.targets.numpy()
        preds = model.predict_proba(test_data).numpy()

        scores = evaluator.score_proba(test_labels, preds)
        for metric in scores:
            results[metric][indx] = scores[metric]

        indx += 1

    output_clf = {}
    output_clf_str = {}

    for key in results:
        key_out = generate_score(results[key])
        output_clf[key] = key_out
        output_clf_str[key] = print_score(key_out)

    return {
        "raw": output_clf,
        "str": output_clf_str,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_regression(
    estimator: Any,
    data: dataset.Dataset,
    n_splits: int = 3,
    seed: int = 0,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    """Helper for evaluating regression tasks.

    Args:
        estimator:
            Baseline model to evaluate. Must not be fitted.
        X: pd.DataFrame or np.ndarray
            covariates
        Y: pd.Series or np.ndarray or list
            outcomes
        n_splits: int
            Number of cross-validation folds
        seed: int
            Random seed

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
        Both "raw" and "str" nodes contain the following metrics:
            - "r2": R^2(coefficient of determination) regression score function.
            - "mse": Mean squared error regression loss.
            - "mae": Mean absolute error regression loss.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    enable_reproducibility(seed)
    metrics = regression_supported_metrics

    metrics_ = {}
    for metric in metrics:
        metrics_[metric] = np.zeros(n_splits)

    indx = 0

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_data, test_data in data.split(splitter=splitter):
        model = copy.deepcopy(estimator)
        model.fit(train_data)

        if test_data.predictive is None:
            raise ValueError("Missing targets for evaluation")

        targets = test_data.predictive.targets.numpy().squeeze()
        preds = model.predict(test_data).numpy().squeeze()

        metrics_["mse"][indx] = mean_squared_error(targets, preds)
        metrics_["mae"][indx] = mean_absolute_error(targets, preds)
        metrics_["r2"][indx] = r2_score(targets, preds)

        indx += 1

    output_mse = generate_score(metrics_["mse"])
    output_mae = generate_score(metrics_["mae"])
    output_r2 = generate_score(metrics_["r2"])

    return {
        "raw": {
            "mse": output_mse,
            "mae": output_mae,
            "r2": output_r2,
        },
        "str": {
            "mse": print_score(output_mse),
            "mae": print_score(output_mae),
            "r2": print_score(output_r2),
        },
    }
