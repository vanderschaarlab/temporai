from typing import Any

import tempor.methods.core as methods_core


class BaseScaler(methods_core.BaseTransformer):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)
