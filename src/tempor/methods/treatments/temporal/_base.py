import abc
from typing import Any, List

import pydantic
from typing_extensions import Self

import tempor.methods.core as methods_core
from tempor.core import pydantic_utils
from tempor.data import dataset, samples


def check_data_class(data: Any) -> None:
    """Check that the passed data is of the correct class (`dataset.OneOffTreatmentEffectsDataset`).

    Args:
        data (Any): Data to check.

    Raises:
        TypeError: If the data is not of the correct class.
    """
    if not isinstance(data, dataset.TemporalTreatmentEffectsDataset):
        raise TypeError(
            "Expected `data` passed to a temporal treatment effects estimator to be "
            f"`{dataset.TemporalTreatmentEffectsDataset.__name__}` but was {type(data)}"
        )


class BaseTemporalTreatmentEffects(methods_core.BasePredictor):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation  # noqa: D107
        super().__init__(**params)

    def fit(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> Self:  # noqa: D102
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def predict(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> samples.TimeSeriesSamples:  # noqa: D102
        check_data_class(data)
        return super().predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(
        self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any
    ) -> samples.TimeSeriesSamples:  # pragma: no cover
        ...

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def predict_counterfactuals(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> List:
        """Predict counterfactuals for the given data.

        Args:
            data (dataset.PredictiveDataset): Data to predict counterfactuals for.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List: List of counterfactual predictions.
        """
        check_data_class(data)
        return super().predict_counterfactuals(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict_counterfactuals(
        self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any
    ) -> List:  # pragma: no cover
        ...
