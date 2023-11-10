from typing import Any

from typing_extensions import Literal

import tempor.methods.core as methods_core

TabularImputerType = Literal[
    "mean",
    "missforest",
    "EM",
    "hyperimpute",
    "sinkhorn",
    "median",
    "sklearn_missforest",
    "miracle",
    "softimpute",
    "mice",
    "gain",
    "sklearn_ice",
    "miwae",
    "ice",
    "most_frequent",
]


class BaseImputer(methods_core.BaseTransformer):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation  # noqa: D107
        super().__init__(**params)
