import abc
from typing import Tuple

import numpy as np
import pydantic
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset, samples


def check_data_class(data):
    if not isinstance(data, dataset.OneOffPredictionDataset):
        raise TypeError(
            "Expected `data` passed to a one-off classification estimator to be "
            f"`{dataset.OneOffPredictionDataset.__name__}` but was {type(data)}"
        )


class BaseOneOffClassifier(plugins.BasePredictor):
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
    ) -> samples.StaticSamples:
        check_data_class(data)
        return super().predict(data, *args, **kwargs)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_proba(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.StaticSamples:
        check_data_class(data)
        return super().predict_proba(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: dataset.Dataset, *args, **kwargs) -> samples.StaticSamples:
        ...

    @abc.abstractmethod
    def _predict_proba(self, data: dataset.Dataset, *args, **kwargs) -> samples.StaticSamples:
        ...

    def _unpack_dataset(self, data: dataset.Dataset) -> Tuple:
        temporal = data.time_series.numpy()
        observation_times = np.asarray(data.time_series.time_indexes())
        if data.predictive is not None:
            outcome = data.predictive.targets.numpy()
        else:
            outcome = np.zeros((len(temporal), 0))

        if data.static is not None:
            static = data.static.numpy()
        else:
            static = np.zeros((len(temporal), 0))

        if len(outcome.shape) == 1:
            outcome = outcome.reshape(-1, 1)
        return static, temporal, observation_times, outcome


plugins.register_plugin_category("prediction.one_off.classification", BaseOneOffClassifier)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseOneOffClassifier",
]
