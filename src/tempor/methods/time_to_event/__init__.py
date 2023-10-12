import abc

import pydantic
from typing_extensions import Any, Self

import tempor.exc
import tempor.methods.core as methods_core
from tempor.core import plugins, pydantic_utils
from tempor.data import data_typing, dataset, samples


def check_data_class(data: Any) -> None:
    if not isinstance(data, dataset.TimeToEventAnalysisDataset):
        raise TypeError(
            "Expected `data` passed to a survival analysis estimator to be "
            f"`{dataset.TimeToEventAnalysisDataset.__name__}` but was {type(data)}"
        )


class BaseTimeToEventAnalysis(methods_core.BasePredictor):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> Self:
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    # NOTE:
    # It appears that `pydantic.validate_arguments` throws an error when `*args: Any` and `**kwargs: Any` are
    # specified here for unknown reasons. For now, we just ignore the type checking for these arguments with
    # `# type: ignore [no-untyped-def]`.
    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def predict(  # type: ignore [no-untyped-def, override] # pylint: disable=arguments-differ
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:  # Output is risk scores at time points, hence `samples.TimeSeriesSamples`.
        check_data_class(data)
        return super().predict(data, horizons, *args, **kwargs)

    def predict_proba(self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:
        raise tempor.exc.UnsupportedSetupException(
            "`predict_proba` method is not supported in the time-to-event analysis setting"
        )

    @abc.abstractmethod
    def _predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self, data: dataset.PredictiveDataset, horizons: data_typing.TimeIndex, *args: Any, **kwargs: Any
    ) -> samples.TimeSeriesSamples:  # pragma: no cover
        ...


plugins.register_plugin_category("time_to_event", BaseTimeToEventAnalysis)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseTimeToEventAnalysis",
]
