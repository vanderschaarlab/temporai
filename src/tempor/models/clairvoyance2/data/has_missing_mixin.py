# mypy: ignore-errors

from typing import Union

import numpy as np
import pandas as pd

from ..utils.common import isnan
from ..utils.dev import raise_not_implemented

TMissingIndicator = Union[float]  # pyright: ignore


# TODO: Unit test.
class HasMissingMixin:
    def __init__(self, missing_indicator: TMissingIndicator = np.nan) -> None:
        self._data: pd.DataFrame
        self.missing_indicator = missing_indicator

        if not isnan(self.missing_indicator):
            raise_not_implemented("Non-nan missing indicators")

    @property
    def has_missing(self) -> bool:
        return bool(self._data.isnull().sum().sum() > 0)  # numpy.bool_ --> bool
