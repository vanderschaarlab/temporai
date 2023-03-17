import abc
from typing import TYPE_CHECKING, Any, Optional

import rich.pretty

from tempor.core.utils import RichReprStrPassthrough

from . import data_typing, samples

if TYPE_CHECKING:  # pragma: no cover
    from .dataset import Dataset  # For typing only, no circular import.


# TODO: Unit test.


class PredictiveTaskData(abc.ABC):
    _targets: samples.DataSamples
    _treatments: Optional[samples.DataSamples]

    @property
    @abc.abstractmethod
    def predictive_task(self) -> data_typing.PredictiveTask:  # pragma: no cover
        ...

    def __init__(
        self,
        parent_dataset: "Dataset",
        targets: Any,  # pylint: disable=unused-argument
        treatments: Optional[Any],  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:  # pragma: no cover
        self.parent_dataset = parent_dataset
        # ^ In order to be able to call parent dataset's `validate` method in the targets/treatments property setters.

        self._targets = targets
        self._treatments = treatments

    def __rich_repr__(self):
        yield "targets", RichReprStrPassthrough(self.targets.short_repr())
        if self.treatments is not None:
            yield "treatments", RichReprStrPassthrough(self.treatments.short_repr())

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    @property
    def targets(self) -> samples.DataSamples:
        return self._targets

    @targets.setter
    def targets(self, value: samples.DataSamples) -> None:
        self._targets = value
        self.parent_dataset.validate()

    @property
    def treatments(self) -> Optional[samples.DataSamples]:
        return self._treatments

    @treatments.setter
    def treatments(self, value: Optional[samples.DataSamples]) -> None:
        self._treatments = value
        self.parent_dataset.validate()


# Predictive task data classes corresponding to different tasks follow. More can be added to handle new tasks.

# --- Prediction tasks: ---


class OneOffPredictionTaskData(PredictiveTaskData):
    # One-off prediction (e.g., one-off classification with a target like patient death).

    targets: samples.StaticSamples
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.ONE_OFF_PREDICTION

    def __init__(self, parent_dataset: "Dataset", targets: data_typing.DataContainer, **kwargs) -> None:
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=None)
        self._targets = samples.StaticSamples(targets)
        self._treatments = None


class TemporalPredictionTaskData(PredictiveTaskData):
    # Temporal prediction (e.g., predicting a patient's temperature real valued time series).

    targets: samples.TimeSeriesSamples
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_PREDICTION

    def __init__(self, parent_dataset: "Dataset", targets: data_typing.DataContainer, **kwargs) -> None:
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=None)
        self._targets = samples.TimeSeriesSamples(targets)
        self._treatments = None


# --- Time-to-event tasks: ---


class TimeToEventAnalysisTaskData(PredictiveTaskData):
    # Time-to-event (survival) analysis (e.g. Dynamic DeepHit).

    targets: samples.EventSamples
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TIME_TO_EVENT_ANALYSIS

    def __init__(self, parent_dataset: "Dataset", targets: data_typing.DataContainer, **kwargs) -> None:
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=None)
        self._targets = samples.EventSamples(targets)
        self._treatments = None


# --- Treatment Effects tasks: ---


class OneOffTreatmentEffectsTaskData(PredictiveTaskData):
    # Treatment effects with time series outcomes but one-off treatment event(s) (e.g. SyncTwin)

    targets: samples.TimeSeriesSamples
    treatments: samples.EventSamples

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS

    def __init__(
        self,
        parent_dataset: "Dataset",
        targets: data_typing.DataContainer,
        treatments: data_typing.DataContainer,
        **kwargs,
    ) -> None:
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=treatments)
        self._targets = samples.TimeSeriesSamples(targets)
        self._treatments = samples.EventSamples(treatments)


class TemporalTreatmentEffectsTaskData(PredictiveTaskData):
    # Temporal treatment effects (i.e. outcomes are time series and treatments are also time series, e.g. RMSN, CRN).

    targets: samples.TimeSeriesSamples
    treatments: samples.TimeSeriesSamples

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS

    def __init__(
        self,
        parent_dataset: "Dataset",
        targets: data_typing.DataContainer,
        treatments: data_typing.DataContainer,
        **kwargs,
    ) -> None:
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=treatments)
        self._targets = samples.TimeSeriesSamples(targets)
        self._treatments = samples.TimeSeriesSamples(treatments)
