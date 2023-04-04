import abc
from typing import List

import pydantic
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset, samples


def check_data_class(data):
    if not isinstance(data, dataset.TemporalTreatmentEffectsDataset):
        raise TypeError(
            "Expected `data` passed to a temporal treatment effects estimator to be "
            f"`{dataset.TemporalTreatmentEffectsDataset.__name__}` but was {type(data)}"
        )


class BaseTemporalTreatmentEffects(plugins.BasePredictor):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.BaseDataset, *args, **kwargs) -> Self:
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> samples.StaticSamples:
        check_data_class(data)
        return super().predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: dataset.BaseDataset, *args, **kwargs) -> samples.StaticSamples:
        ...

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_counterfactuals(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> List:
        check_data_class(data)
        return super().predict_counterfactuals(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict_counterfactuals(self, data: dataset.BaseDataset, *args, **kwargs) -> List:
        ...
