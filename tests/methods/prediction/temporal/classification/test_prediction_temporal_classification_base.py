from typing import TYPE_CHECKING, Any, List

import pytest

from tempor.data.dataset import TemporalPredictionDataset
from tempor.methods.prediction.temporal.classification import BaseTemporalClassifier

if TYPE_CHECKING:
    from typing_extensions import Self
    from tempor.methods.core import Params
    from tempor.data.dataset import BaseDataset, PredictiveDataset
    from tempor.data.samples import TimeSeriesSamples

from unittest.mock import MagicMock, Mock

mock_predict_proba = Mock()


class DummyTemporalClassifier(BaseTemporalClassifier):
    name = "dummy"
    category = "dummy_cat"
    plugin_type = "dummy_type"

    def _predict(
        self,
        data: "PredictiveDataset",
        n_future_steps: int,
        *args,
        time_delta: int = 1,
        **kwargs,
    ) -> "TimeSeriesSamples":
        raise NotImplementedError

    def _predict_proba(
        self,
        data: "PredictiveDataset",
        n_future_steps: int,
        *args,
        time_delta: int = 1,
        **kwargs,
    ) -> "TimeSeriesSamples":
        mock_predict_proba()
        return Mock()

    def _fit(self, data: "BaseDataset", *args, **kwargs) -> "Self":
        return super()._fit(data, *args, **kwargs)

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List["Params"]:
        raise NotImplementedError


def test_wrong_dataset():
    plugin = DummyTemporalClassifier()

    with pytest.raises(TypeError, match="[Ee]xpected.*data.*temporal classification.*"):
        plugin.fit(data=Mock())


def test_predict_proba():
    plugin = DummyTemporalClassifier()
    plugin._fitted = True  # pylint: disable=protected-access

    plugin.predict_proba(MagicMock(TemporalPredictionDataset), n_future_steps=1)
    mock_predict_proba.assert_called_once()
