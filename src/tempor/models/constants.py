"""Common constants for the ``models`` package directory."""

import torch
from typing_extensions import Literal

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""`torch.device` to use for computation, defaults to ``"cuda:0"`` if available, else ``"cpu"``."""

EPS: float = torch.finfo(torch.float32).eps
"""Machine epsilon for ``torch.float32``."""

ModelTaskType = Literal["classification", "regression"]
"""Possible values specifying a model task type."""

ODEBackend = Literal["ode", "cde", "laplace"]
"""Possible values specifying an ODE backend."""

Nonlin = Literal[
    "none",
    "elu",
    "relu",
    "leaky_relu",
    "selu",
    "tanh",
    "sigmoid",
    "softmax",
    "gumbel_softmax",
]
"""Possible values specifying a nonlinearity."""

Samp = Literal[
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]
"""Possible values specifying a Sampler."""
