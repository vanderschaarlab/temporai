from typing import TYPE_CHECKING, Any, List

import pytest

from tempor.methods.treatments.temporal import BaseTemporalTreatmentEffects

if TYPE_CHECKING:
    from typing_extensions import Self
    from tempor.methods.core._params import Params
    from tempor.data.dataset import BaseDataset, PredictiveDataset
    from tempor.data.samples import TimeSeriesSamples

from unittest.mock import Mock


class DummyTemporalTreatmentEffects(BaseTemporalTreatmentEffects):
    name = "dummy"
    category = "dummy_cat"
    plugin_type = "dummy_type"

    def _predict(
        self,
        data: "PredictiveDataset",
        *args,
        **kwargs,
    ) -> "TimeSeriesSamples":
        return Mock()

    def _predict_counterfactuals(self, data: "PredictiveDataset", *args, **kwargs) -> List:
        return []

    def _fit(self, data: "BaseDataset", *args, **kwargs) -> "Self":
        return self

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List["Params"]:
        raise NotImplementedError


def test_wrong_dataset():
    plugin = DummyTemporalTreatmentEffects()

    with pytest.raises(TypeError, match="[Ee]xpected.*data.*temporal treatment effects.*"):
        plugin.fit(data=Mock())
