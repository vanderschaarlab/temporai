# pylint: disable=unused-argument

import abc

import pydantic
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset, samples


def check_data_class(data):
    if not isinstance(data, dataset.TemporalPredictionDataset):
        raise TypeError(
            "Expected `data` passed to a temporal classification estimator to be "
            f"`{dataset.TemporalPredictionDataset.__name__}` but was {type(data)}"
        )


class BaseTemporalClassifier(plugins.BasePredictor):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.BaseDataset, *args, **kwargs) -> Self:
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.BaseDataset,
        n_future_steps: int,
        *args,
        time_delta: int = 1,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        check_data_class(data)
        return super().predict(data, n_future_steps, *args, time_delta=time_delta, **kwargs)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_proba(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.BaseDataset,
        n_future_steps: int,
        *args,
        time_delta: int = 1,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        check_data_class(data)
        return super().predict_proba(data, n_future_steps, *args, time_delta=time_delta, **kwargs)

    @abc.abstractmethod
    def _predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, data: dataset.BaseDataset, n_future_steps: int, *args, time_delta: int = 1, **kwargs
    ) -> samples.TimeSeriesSamples:
        ...

    @abc.abstractmethod
    def _predict_proba(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, data: dataset.BaseDataset, n_future_steps: int, *args, time_delta: int = 1, **kwargs
    ) -> samples.TimeSeriesSamples:
        ...


plugins.register_plugin_category("prediction.temporal.classification", BaseTemporalClassifier)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseTemporalClassifier",
]
