from typing import TYPE_CHECKING, Any, List

import pytest

from tempor.plugins.prediction.temporal.regression import BaseTemporalRegressor

if TYPE_CHECKING:
    from typing_extensions import Self
    from tempor.plugins.core._params import Params
    from tempor.data.dataset import BaseDataset, PredictiveDataset
    from tempor.data.samples import TimeSeriesSamples

from unittest.mock import Mock


class DummyTemporalRegressor(BaseTemporalRegressor):
    name = "dummy"
    category = "dummy_cat"

    def _predict(
        self,
        data: "PredictiveDataset",
        n_future_steps: int,
        *args,
        time_delta: int = 1,
        **kwargs,
    ) -> "TimeSeriesSamples":
        raise NotImplementedError

    def _fit(self, data: "BaseDataset", *args, **kwargs) -> "Self":
        return super()._fit(data, *args, **kwargs)

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List["Params"]:
        raise NotImplementedError


def test_wrong_dataset():
    plugin = DummyTemporalRegressor()

    with pytest.raises(TypeError, match="[Ee]xpected.*data.*temporal regression.*"):
        plugin.fit(data=Mock())
