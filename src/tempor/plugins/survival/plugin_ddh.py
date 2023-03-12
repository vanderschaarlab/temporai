import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.plugins.survival import BaseSurvivalAnalysis


@plugins.register_plugin(name="dynamic_deephit", category="survival")
class DynamicDeepHitSurvivalAnalysis(BaseSurvivalAnalysis):
    def __init__(self, **params) -> None:  # pylint: disable=E,W
        raise NotImplementedError

    def fit(  # type: ignore[override]
        self,
        data: dataset.TimeToEventAnalysisDataset,
        *args,
        **kwargs,
    ) -> "DynamicDeepHitSurvivalAnalysis":  # pyright: ignore
        return super().fit(data, *args, **kwargs)  # type: ignore[return-value]

    def _fit(  # type: ignore[override]
        self,
        data: dataset.TimeToEventAnalysisDataset,
        *args,
        **kwargs,
    ) -> "DynamicDeepHitSurvivalAnalysis":  # pyright: ignore
        raise NotImplementedError

    def _predict(  # type: ignore[override]
        self,
        data: dataset.TimeToEventAnalysisDataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        raise NotImplementedError
