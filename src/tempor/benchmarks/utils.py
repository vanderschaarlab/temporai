"""Any utilities for the ``benchmark`` package directory."""

from typing import Tuple

import numpy as np


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
