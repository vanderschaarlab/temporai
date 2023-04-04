import dataclasses
from typing import Any, Tuple
from unittest.mock import MagicMock, Mock

import pytest

import tempor.plugins.core
from tempor.data.dataset import PredictiveDataset


class TestBasePredictor:
    @dataclasses.dataclass
    class MyModelParams:
        foo: float = 1.5
        bar: Tuple[int, ...] = (3, 7, 9)
        baz: Any = "something"

    def test_predict(self):
        class MyModel(tempor.plugins.core.BasePredictor):
            name = "my_model"
            category = "my_category"
            ParamsDefinition = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

            def _predict(self, data: PredictiveDataset, *args, **kwargs) -> Any:
                return super()._predict(data, *args, **kwargs)

        my_model = MyModel()
        data = Mock(spec=PredictiveDataset)
        data.predict_ready = True

        my_model.predict(data)

    def test_predict_fails_data_not_fit_ready(self):
        class MyModel(tempor.plugins.core.BasePredictor):
            name = "my_model"
            category = "my_category"
            ParamsDefinition = self.MyModelParams

            @staticmethod
            def hyperparameter_space(*args: Any, **kwargs: Any):
                return MagicMock()

            def _fit(self, data, *args, **kwargs):
                pass

            def _predict(self, data: PredictiveDataset, *args, **kwargs) -> Any:
                pass

            def _predict_proba(self, data: PredictiveDataset, *args, **kwargs) -> Any:
                pass

            def _predict_counterfactuals(self, data: PredictiveDataset, *args, **kwargs) -> Any:
                pass

        my_model = MyModel()
        data = Mock(spec=PredictiveDataset)
        data.predict_ready = False

        with pytest.raises(ValueError, match=".*not predict-ready.*"):
            my_model.predict(data)

        with pytest.raises(ValueError, match=".*not predict-ready.*"):
            my_model.predict_proba(data)

        with pytest.raises(ValueError, match=".*not predict-ready.*"):
            my_model.predict_counterfactuals(data)
