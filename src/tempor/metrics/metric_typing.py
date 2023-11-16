"""Typing related to metrics."""

from typing import Tuple

import numpy as np
from typing_extensions import Literal

MetricDirection = Literal["minimize", "maximize"]
"""The direction of the metric that represents the optimization goal (the "good" direction):
``"minimize"`` or "``maximize``".
"""

EventArrayTimeArray = Tuple[np.ndarray, np.ndarray]
"""Type hint used in time-to-event metrics, a tuple of two numpy arrays:
the event values array and the event times array.
"""
