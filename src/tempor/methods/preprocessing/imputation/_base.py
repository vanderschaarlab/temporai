from typing_extensions import Literal

import tempor.methods.core as plugins

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


class BaseImputer(plugins.BaseTransformer):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)
