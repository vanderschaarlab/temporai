import abc
from typing import Tuple

import numpy as np
import pydantic
from typing_extensions import Self

import tempor.methods.core as plugins
from tempor.data import dataset, samples


def check_data_class(data):
    if not isinstance(data, dataset.OneOffPredictionDataset):
        raise TypeError(
            "Expected `data` passed to a one-off regression estimator to be "
            f"`{dataset.OneOffPredictionDataset.__name__}` but was {type(data)}"
        )


class BaseOneOffRegressor(plugins.BasePredictor):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.BaseDataset, *args, **kwargs) -> Self:
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))  # type: ignore [operator]
    def predict(
        self,
        data: dataset.PredictiveDataset,
        *args,
        **kwargs,
    ) -> samples.StaticSamples:
        check_data_class(data)
        return super().predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: dataset.PredictiveDataset, *args, **kwargs) -> samples.StaticSamples:  # pragma: no cover
        ...

    def _unpack_dataset(self, data: dataset.BaseDataset) -> Tuple:
        temporal = data.time_series.numpy()
        observation_times = data.time_series.time_indexes()
        if data.predictive is not None and data.predictive.targets is not None:
            outcome = data.predictive.targets.numpy()
        else:
            outcome = np.zeros((len(temporal), 0))

        if data.static is not None:
            static = data.static.numpy()
        else:
            static = np.zeros((len(temporal), 0))

        return static, temporal, observation_times, outcome


plugins.register_plugin_category("prediction.one_off.regression", BaseOneOffRegressor)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseOneOffRegressor",
]
