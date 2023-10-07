from typing import TYPE_CHECKING, Any, List

import pytest

from tempor.data.dataset import TimeToEventAnalysisDataset
from tempor.exc import UnsupportedSetupException
from tempor.plugins.time_to_event import BaseTimeToEventAnalysis

if TYPE_CHECKING:
    from typing_extensions import Self
    from tempor.plugins.core._params import Params
    from tempor.data.dataset import BaseDataset, PredictiveDataset
    from tempor.data.samples import TimeSeriesSamples

from unittest.mock import MagicMock, Mock

mock_predict_proba = Mock()


class DummyTimeToEventAnalysis(BaseTimeToEventAnalysis):
    name = "dummy"
    category = "dummy_cat"
    plugin_type = "dummy_type"

    def _predict(
        self,
        data: "PredictiveDataset",
        horizons,
        *args,
        **kwargs,
    ) -> "TimeSeriesSamples":
        raise NotImplementedError

    def _fit(self, data: "BaseDataset", *args, **kwargs) -> "Self":
        return super()._fit(data, *args, **kwargs)

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List["Params"]:
        raise NotImplementedError


def test_wrong_dataset():
    plugin = DummyTimeToEventAnalysis()

    with pytest.raises(TypeError, match="[Ee]xpected.*data.*survival analysis.*"):
        plugin.fit(data=Mock())


def test_predict_proba():
    plugin = DummyTimeToEventAnalysis()
    plugin._fitted = True  # pylint: disable=protected-access

    with pytest.raises(UnsupportedSetupException):
        plugin.predict_proba(MagicMock(TimeToEventAnalysisDataset))
