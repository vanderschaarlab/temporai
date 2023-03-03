import abc
from typing import Any, Optional

import rich.pretty

from tempor.core.utils import RichReprStrPassthrough

from . import data_typing, samples

# TODO: Unit test.


class PredictiveTaskData(abc.ABC):
    targets: samples.DataSamples
    treatments: Optional[samples.DataSamples]

    @property
    @abc.abstractmethod
    def predictive_task(self) -> data_typing.PredictiveTask:  # pragma: no cover
        ...

    @abc.abstractmethod
    def __init__(
        self,
        targets: Any,
        treatments: Optional[Any],
        **kwargs,
    ) -> None:  # pragma: no cover
        ...

    def __rich_repr__(self):
        yield "targets", RichReprStrPassthrough(self.targets.short_repr())
        if self.treatments is not None:
            yield "treatments", RichReprStrPassthrough(self.treatments.short_repr())

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)


# Predictive task data classes corresponding to different tasks follow. More can be added to handle new tasks.

# --- Prediction tasks: ---


class OneOffPredictionTaskData(PredictiveTaskData):
    # One-off prediction (e.g., one-off classification with a target like patient death).

    targets: samples.StaticSamples
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.ONE_OFF_PREDICTION

    def __init__(self, targets: data_typing.DataContainer, **kwargs) -> None:
        self.targets = samples.StaticSamples(targets)
        self.treatments = None
        super().__init__(targets=targets, treatments=None)


class TemporalPredictionTaskData(PredictiveTaskData):
    # Temporal prediction (e.g., predicting a patient's temperature real valued time series).

    targets: samples.TimeSeriesSamples
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_PREDICTION

    def __init__(self, targets: data_typing.DataContainer, **kwargs) -> None:
        self.targets = samples.TimeSeriesSamples(targets)
        self.treatments = None
        super().__init__(targets=targets, treatments=None)


# --- Time-to-event tasks: ---


class TimeToEventAnalysisTaskData(PredictiveTaskData):
    # Time-to-event (survival) analysis (e.g. Dynamic DeepHit).

    targets: samples.EventSamples
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TIME_TO_EVENT_ANALYSIS

    def __init__(self, targets: data_typing.DataContainer, **kwargs) -> None:
        self.targets = samples.EventSamples(targets)
        self.treatments = None
        super().__init__(targets=targets, treatments=None)


# --- Treatment Effects tasks: ---


class OneOffTreatmentEffectsTaskData(PredictiveTaskData):
    # Treatment effects with time series outcomes but one-off treatment event(s) (e.g. SyncTwin)

    targets: samples.TimeSeriesSamples
    treatments: samples.EventSamples

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS

    def __init__(self, targets: data_typing.DataContainer, treatments: data_typing.DataContainer, **kwargs) -> None:
        self.targets = samples.TimeSeriesSamples(targets)
        self.treatments = samples.EventSamples(treatments)
        super().__init__(targets=targets, treatments=treatments)


class TemporalTreatmentEffectsTaskData(PredictiveTaskData):
    # Temporal treatment effects (i.e. outcomes are time series and treatments are also time series, e.g. RMSN, CRN).

    targets: samples.TimeSeriesSamples
    treatments: samples.TimeSeriesSamples

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS

    def __init__(self, targets: data_typing.DataContainer, treatments: data_typing.DataContainer, **kwargs) -> None:
        self.targets = samples.TimeSeriesSamples(targets)
        self.treatments = samples.TimeSeriesSamples(treatments)
        super().__init__(targets=targets, treatments=treatments)
