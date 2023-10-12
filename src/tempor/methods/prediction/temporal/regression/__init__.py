import abc
from typing import Any

import pydantic
from typing_extensions import Self

import tempor.methods.core as methods_core
from tempor.core import plugins
from tempor.data import dataset, samples


def check_data_class(data: Any) -> None:
    if not isinstance(data, dataset.TemporalPredictionDataset):
        raise TypeError(
            "Expected `data` passed to a temporal regression estimator to be "
            f"`{dataset.TemporalPredictionDataset.__name__}` but was {type(data)}"
        )


class BaseTemporalRegressor(methods_core.BasePredictor):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> Self:
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))  # type: ignore [operator]
    def predict(  # pylint: disable=arguments-differ
        self,
        data: dataset.PredictiveDataset,
        n_future_steps: int,
        *args: Any,
        time_delta: int = 1,
        **kwargs: Any,
    ) -> samples.TimeSeriesSamples:
        check_data_class(data)
        return super().predict(data, n_future_steps, *args, time_delta=time_delta, **kwargs)

    @abc.abstractmethod
    def _predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, data: dataset.PredictiveDataset, n_future_steps: int, *args: Any, time_delta: int = 1, **kwargs: Any
    ) -> samples.TimeSeriesSamples:  # pragma: no cover
        ...


plugins.register_plugin_category("prediction.temporal.regression", BaseTemporalRegressor)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseTemporalRegressor",
]
