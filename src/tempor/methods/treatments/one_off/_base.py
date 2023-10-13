import abc
from typing import Any, List

import pydantic
from typing_extensions import Self

import tempor.methods.core as methods_core
from tempor.core import pydantic_utils
from tempor.data import dataset, samples


def check_data_class(data: Any) -> None:
    if not isinstance(data, dataset.OneOffTreatmentEffectsDataset):
        raise TypeError(
            "Expected `data` passed to a one-off treatment effects estimator to be "
            f"`{dataset.OneOffTreatmentEffectsDataset.__name__}` but was {type(data)}"
        )


class BaseOneOffTreatmentEffects(methods_core.BasePredictor):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> Self:
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def predict(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> samples.StaticSamples:
        check_data_class(data)
        return super().predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(
        self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any
    ) -> samples.StaticSamples:  # pragma: no cover
        ...

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def predict_counterfactuals(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> List:
        check_data_class(data)
        return super().predict_counterfactuals(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict_counterfactuals(
        self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any
    ) -> List:  # pragma: no cover
        ...
