"""Module defining the `PredictiveTaskData` class and its subclasses, which are used to store the data components
relevant for different predictive tasks (e.g. prediction, time-to-event analysis, treatment effects).
"""

import abc
from typing import TYPE_CHECKING, Any, Generator, Optional

import rich.pretty

from tempor.core.utils import RichReprStrPassthrough

from . import data_typing, samples

if TYPE_CHECKING:  # pragma: no cover
    from .dataset import PredictiveDataset  # For typing only, no circular import.


class PredictiveTaskData(abc.ABC):
    _targets: Optional[samples.DataSamples]
    _treatments: Optional[samples.DataSamples]

    @property
    @abc.abstractmethod
    def predictive_task(self) -> data_typing.PredictiveTask:  # pragma: no cover
        """Return the predictive task enum value corresponding to the class.

        Returns:
            data_typing.PredictiveTask: The predictive task enum value.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def __init__(  # pylint: disable=unused-argument
        self,
        parent_dataset: "PredictiveDataset",
        targets: Any,
        treatments: Optional[Any],
        **kwargs: Any,
    ) -> None:
        """The predictive task data abstract base class.

        Args:
            parent_dataset (PredictiveDataset): The parent predictive dataset object.
            targets (Any): The targets data.
            treatments (Optional[Any]): The treatments data.
            **kwargs (Any): Additional keyword arguments. Currently unused.
        """
        self.parent_dataset = parent_dataset
        # ^ In order to be able to call parent dataset's `validate` method in the targets/treatments property setters.

        self._targets = targets
        self._treatments = treatments

    def __rich_repr__(self) -> Generator:
        """Representation for `rich`.

        Yields:
            Generator: Fields and their values for `rich`.
        """
        if self.targets is not None:
            yield "targets", RichReprStrPassthrough(self.targets.short_repr())
        else:
            yield "targets", None
        if self.treatments is not None:
            yield "treatments", RichReprStrPassthrough(self.treatments.short_repr())

    def __repr__(self) -> str:
        """Representation for `repr()`.

        Returns:
            str: The representation string.
        """
        return rich.pretty.pretty_repr(self)

    @property
    def targets(self) -> Optional[samples.DataSamples]:
        """The property containing the targets data.

        Returns:
            Optional[samples.DataSamples]: The targets data.
        """
        return self._targets

    @targets.setter
    def targets(self, value: Optional[samples.DataSamples]) -> None:
        self._targets = value
        self.parent_dataset.validate()

    @property
    def treatments(self) -> Optional[samples.DataSamples]:
        """The property containing the treatments data.

        Returns:
            Optional[samples.DataSamples]: The treatments data.
        """
        return self._treatments

    @treatments.setter
    def treatments(self, value: Optional[samples.DataSamples]) -> None:
        self._treatments = value
        self.parent_dataset.validate()


# Predictive task data classes corresponding to different tasks follow. More can be added to handle new tasks.

# --- Prediction tasks: ---


class OneOffPredictionTaskData(PredictiveTaskData):
    # One-off prediction (e.g., one-off classification with a target like patient death).

    targets: Optional[samples.StaticSamples]
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """Return the predictive task enum value corresponding to the class. Here, ``ONE_OFF_PREDICTION``.

        Returns:
            data_typing.PredictiveTask: The predictive task enum value. Here, ``ONE_OFF_PREDICTION``.
        """
        return data_typing.PredictiveTask.ONE_OFF_PREDICTION

    def __init__(
        self, parent_dataset: "PredictiveDataset", targets: Optional[data_typing.DataContainer], **kwargs: Any
    ) -> None:
        """The one-off prediction task data class.

        Args:
            parent_dataset (PredictiveDataset): The parent predictive dataset object.
            targets (Optional[data_typing.DataContainer]): The targets data.
            **kwargs (Any): Additional keyword arguments. Currently unused.
        """
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=None)
        self._targets = samples.StaticSamples(targets) if targets is not None else None
        self._treatments = None


class TemporalPredictionTaskData(PredictiveTaskData):
    # Temporal prediction (e.g., predicting a patient's temperature real valued time series).

    targets: Optional[samples.TimeSeriesSamples]
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """Return the predictive task enum value corresponding to the class. Here, ``TEMPORAL_PREDICTION``.

        Returns:
            data_typing.PredictiveTask: The predictive task enum value. Here, ``TEMPORAL_PREDICTION``.
        """
        return data_typing.PredictiveTask.TEMPORAL_PREDICTION

    def __init__(
        self, parent_dataset: "PredictiveDataset", targets: Optional[data_typing.DataContainer], **kwargs: Any
    ) -> None:
        """The temporal prediction task data class.

        Args:
            parent_dataset (PredictiveDataset): The parent predictive dataset object.
            targets (Optional[data_typing.DataContainer]): The targets data.
            **kwargs (Any): Additional keyword arguments. Currently unused.
        """
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=None)
        self._targets = samples.TimeSeriesSamples(targets) if targets is not None else None
        self._treatments = None


# --- Time-to-event tasks: ---


class TimeToEventAnalysisTaskData(PredictiveTaskData):
    # Time-to-event (survival) analysis (e.g. Dynamic DeepHit).

    targets: Optional[samples.EventSamples]
    treatments: None

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """Return the predictive task enum value corresponding to the class. Here, ``TIME_TO_EVENT_ANALYSIS``.

        Returns:
            data_typing.PredictiveTask: The predictive task enum value. Here, ``TIME_TO_EVENT_ANALYSIS``.
        """
        return data_typing.PredictiveTask.TIME_TO_EVENT_ANALYSIS

    def __init__(
        self, parent_dataset: "PredictiveDataset", targets: Optional[data_typing.DataContainer], **kwargs: Any
    ) -> None:
        """The time-to-event analysis task data class.

        Args:
            parent_dataset (PredictiveDataset): The parent predictive dataset object.
            targets (Optional[data_typing.DataContainer]): The targets data.
            **kwargs (Any): Additional keyword arguments. Currently unused.
        """
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=None)
        self._targets = samples.EventSamples(targets) if targets is not None else None
        self._treatments = None


# --- Treatment Effects tasks: ---


class OneOffTreatmentEffectsTaskData(PredictiveTaskData):
    # Treatment effects with time series outcomes but one-off treatment event(s) (e.g. SyncTwin)

    targets: Optional[samples.TimeSeriesSamples]
    treatments: samples.EventSamples

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """Return the predictive task enum value corresponding to the class. Here, ``ONE_OFF_TREATMENT_EFFECTS``.

        Returns:
            data_typing.PredictiveTask: The predictive task enum value. Here, ``ONE_OFF_TREATMENT_EFFECTS``.
        """
        return data_typing.PredictiveTask.ONE_OFF_TREATMENT_EFFECTS

    def __init__(
        self,
        parent_dataset: "PredictiveDataset",
        targets: Optional[data_typing.DataContainer],
        treatments: data_typing.DataContainer,
        **kwargs: Any,
    ) -> None:
        """The one-off treatment effects task data class.

        Args:
            parent_dataset (PredictiveDataset): The parent predictive dataset object.
            targets (Optional[data_typing.DataContainer]): The targets data.
            treatments (data_typing.DataContainer): The treatments data.
            **kwargs (Any): Additional keyword arguments. Currently unused.
        """
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=treatments)
        self._targets = samples.TimeSeriesSamples(targets) if targets is not None else None
        self._treatments = samples.EventSamples(treatments)


class TemporalTreatmentEffectsTaskData(PredictiveTaskData):
    # Temporal treatment effects (i.e. outcomes are time series and treatments are also time series, e.g. RMSN, CRN).

    targets: Optional[samples.TimeSeriesSamples]
    treatments: samples.TimeSeriesSamples

    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """Return the predictive task enum value corresponding to the class. Here, ``TEMPORAL_TREATMENT_EFFECTS``.

        Returns:
            data_typing.PredictiveTask: The predictive task enum value. Here, ``TEMPORAL_TREATMENT_EFFECTS``.
        """
        return data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS

    def __init__(
        self,
        parent_dataset: "PredictiveDataset",
        targets: Optional[data_typing.DataContainer],
        treatments: data_typing.DataContainer,
        **kwargs: Any,
    ) -> None:
        """The temporal treatment effects task data class.

        Args:
            parent_dataset (PredictiveDataset): The parent predictive dataset object.
            targets (Optional[data_typing.DataContainer]): The targets data.
            treatments (data_typing.DataContainer): The treatments data.
            **kwargs (Any): Additional keyword arguments. Currently unused.
        """
        super().__init__(parent_dataset=parent_dataset, targets=targets, treatments=treatments)
        self._targets = samples.TimeSeriesSamples(targets) if targets is not None else None
        self._treatments = samples.TimeSeriesSamples(treatments)
