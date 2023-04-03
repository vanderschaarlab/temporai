import abc

import pydantic
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset, samples

# TODO: DIFFERENT PREDICTION SIGNATURE!!!! --------------------------------------------------------------------------------------------


def check_data_class(data):
    if not isinstance(data, dataset.TemporalPredictionDataset):
        raise TypeError(
            "Expected `data` passed to a temporal classification estimator to be "
            f"`{dataset.TemporalPredictionDataset.__name__}` but was {type(data)}"
        )


class BaseTemporalClassifier(plugins.BasePredictor):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.Dataset, *args, **kwargs) -> Self:
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        check_data_class(data)
        return super().predict(data, *args, **kwargs)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_proba(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        check_data_class(data)
        return super().predict_proba(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: dataset.Dataset, *args, **kwargs) -> samples.TimeSeriesSamples:
        ...

    @abc.abstractmethod
    def _predict_proba(self, data: dataset.Dataset, *args, **kwargs) -> samples.TimeSeriesSamples:
        ...


plugins.register_plugin_category("prediction.temporal.classification", BaseTemporalClassifier)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseTemporalClassifier",
]
