from typing import TYPE_CHECKING, Any, List

import numpy as np
import pytest

from tempor.data.dataset import PredictiveDataset
from tempor.methods.prediction.one_off.classification import BaseOneOffClassifier

if TYPE_CHECKING:
    from typing_extensions import Self
    from tempor.methods.core import Params
    from tempor.data.dataset import BaseDataset
    from tempor.data.samples import StaticSamples

from unittest.mock import Mock


class DummyOneOffClassifier(BaseOneOffClassifier):
    name = "dummy"
    category = "dummy_cat"
    plugin_type = "dummy_type"

    def _predict(self, data: "PredictiveDataset", *args, **kwargs) -> "StaticSamples":
        raise NotImplementedError

    def _predict_proba(self, data: "PredictiveDataset", *args, **kwargs) -> "StaticSamples":
        raise NotImplementedError

    def _fit(self, data: "BaseDataset", *args, **kwargs) -> "Self":
        return super()._fit(data, *args, **kwargs)

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List["Params"]:
        raise NotImplementedError


def test_wrong_dataset():
    plugin = DummyOneOffClassifier()

    with pytest.raises(TypeError, match="[Ee]xpected.*data.*one-off classification.*"):
        plugin.fit(data=Mock())


def test_unpack_dataset_empty():
    plugin = DummyOneOffClassifier()

    mock_data = Mock(
        PredictiveDataset,
        time_series=Mock(
            numpy=Mock(return_value=[1, 2, 3]),
            time_indexes=Mock(return_value="mock_time_indexes"),
        ),
        static=None,
        predictive=Mock(
            targets=None,
        ),
    )

    _, _, _, outcome = plugin._unpack_dataset(data=mock_data)  # pylint: disable=protected-access

    assert outcome.shape == np.zeros((3, 0)).shape

    mock_data = Mock(
        PredictiveDataset,
        time_series=Mock(
            numpy=Mock(return_value=[1, 2, 3]),
            time_indexes=Mock(return_value="mock_time_indexes"),
        ),
        static=None,
        predictive=Mock(
            targets=Mock(numpy=Mock(return_value=np.asarray([1, 1, 1, 1]))),
        ),
    )

    _, _, _, outcome = plugin._unpack_dataset(data=mock_data)  # pylint: disable=protected-access

    assert outcome.shape == np.zeros((4, 1)).shape
