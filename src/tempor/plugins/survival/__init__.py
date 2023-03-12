import tempor.plugins.core as plugins
from tempor.data import dataset, samples


class BaseSurvivalAnalysis(plugins.BasePredictor):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def fit(  # type: ignore[override]
        self,
        data: dataset.TimeToEventAnalysisDataset,
        *args,
        **kwargs,
    ) -> "BaseSurvivalAnalysis":
        return super().fit(data, *args, **kwargs)  # type: ignore[return-value]

    def _fit(  # type: ignore[override]
        self,
        data: dataset.TimeToEventAnalysisDataset,
        *args,
        **kwargs,
    ) -> "BaseSurvivalAnalysis":
        return super().fit(data, *args, **kwargs)  # type: ignore[return-value]

    def predict(  # type: ignore[override]
        self,
        data: dataset.TimeToEventAnalysisDataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:  # Output is risk scores at time points, hence `samples.TimeSeriesSamples`.
        return super().predict(data, *args, **kwargs)

    def _predict(  # type: ignore[override]
        self,
        data: dataset.TimeToEventAnalysisDataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        return super()._predict(data, *args, **kwargs)


plugins.register_plugin_category("survival", BaseSurvivalAnalysis)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseSurvivalAnalysis",
]
