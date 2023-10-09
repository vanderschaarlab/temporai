import abc

import pydantic
from typing_extensions import Any, Self

import tempor.core.plugins as plugins
import tempor.exc
import tempor.methods.core as methods_core
from tempor.data import data_typing, dataset, samples


def check_data_class(data):
    if not isinstance(data, dataset.TimeToEventAnalysisDataset):
        raise TypeError(
            "Expected `data` passed to a survival analysis estimator to be "
            f"`{dataset.TimeToEventAnalysisDataset.__name__}` but was {type(data)}"
        )


class BaseTimeToEventAnalysis(methods_core.BasePredictor):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.BaseDataset, *args, **kwargs) -> Self:
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))  # type: ignore [operator]
    def predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:  # Output is risk scores at time points, hence `samples.TimeSeriesSamples`.
        check_data_class(data)
        return super().predict(data, horizons, *args, **kwargs)

    def predict_proba(self, data: dataset.PredictiveDataset, *args, **kwargs) -> Any:
        raise tempor.exc.UnsupportedSetupException(
            "`predict_proba` method is not supported in the time-to-event analysis setting"
        )

    @abc.abstractmethod
    def _predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, data: dataset.PredictiveDataset, horizons: data_typing.TimeIndex, *args, **kwargs
    ) -> samples.TimeSeriesSamples:  # pragma: no cover
        ...


plugins.register_plugin_category("time_to_event", BaseTimeToEventAnalysis)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseTimeToEventAnalysis",
]
