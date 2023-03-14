import abc

import pydantic

import tempor.plugins.core as plugins
from tempor.data import dataset, samples


def check_data_class(data):
    if not isinstance(data, dataset.TimeToEventAnalysisDataset):
        raise TypeError(
            "Expected `data` passed to a survival analysis estimator to be "
            f"`{dataset.TimeToEventAnalysisDataset.__name__}` but was {type(data)}"
        )


class BaseSurvivalAnalysis(plugins.BasePredictor):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(self, data: dataset.Dataset, *args, **kwargs) -> "BaseSurvivalAnalysis":
        check_data_class(data)
        super().fit(data, *args, **kwargs)
        return self

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:  # Output is risk scores at time points, hence `samples.TimeSeriesSamples`.
        check_data_class(data)
        return super().predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: dataset.Dataset, *args, **kwargs) -> samples.TimeSeriesSamples:
        ...


plugins.register_plugin_category("survival", BaseSurvivalAnalysis)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseSurvivalAnalysis",
]
