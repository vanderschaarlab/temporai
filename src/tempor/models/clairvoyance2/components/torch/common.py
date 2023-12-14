# mypy: ignore-errors

from typing import Dict, Mapping, Optional, Type

import torch
import torch.nn as nn

ACTIVATION_MAP: Mapping[str, type] = {
    "ReLU": nn.ReLU,
    "Softmax": nn.Softmax,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}
ACTIVATION_KWARGS: Mapping[str, Mapping] = {
    "ReLU": dict(),
    "Softmax": dict(dim=-1),
    "Sigmoid": dict(),
    "Tanh": dict(),
}
# TODO: ^ Add more.


def init_activation(activation: str, kwargs: Optional[Mapping[str, Dict]] = None) -> nn.Module:
    kwargs = kwargs if kwargs is not None else ACTIVATION_KWARGS[activation]
    return ACTIVATION_MAP[activation](**ACTIVATION_KWARGS[activation])


OPTIM_MAP: Mapping[str, Type[torch.optim.Optimizer]] = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    # TODO: Allow more.
}
