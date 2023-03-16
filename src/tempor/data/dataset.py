# pylint: disable=unnecessary-ellipsis

import abc
import dataclasses
from typing import ClassVar, Generator, Optional, Union

import rich.pretty

from tempor.core.utils import RichReprStrPassthrough
from tempor.log import log_helpers

from . import data_typing
from . import predictive as pred
from . import samples


@dataclasses.dataclass(frozen=True)
class _SampleIndexMismatchMsg:
    static: ClassVar[str] = "`sample_index` of static samples did not match `sample_index` of time series samples"
    targets: ClassVar[str] = "`sample_index` of targets did not match `sample_index` of time series samples"
    treatments: ClassVar[str] = "`sample_index` of treatments did not match `sample_index` of time series samples"


@dataclasses.dataclass(frozen=True)
class _ExceptionMessages:
    sample_index_mismatch: ClassVar[_SampleIndexMismatchMsg] = _SampleIndexMismatchMsg()


EXCEPTION_MESSAGES = _ExceptionMessages()
"""Reusable error messages for the module."""


class Dataset(abc.ABC):
    time_series: samples.TimeSeriesSamples
    static: Optional[samples.StaticSamples]
    predictive: Optional[pred.PredictiveTaskData]

    def __init__(
        self,
        time_series: data_typing.DataContainer,
        *,
        static: Optional[data_typing.DataContainer] = None,
        targets: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """Base class representing a dataset used by TemporAI.

        Initialize one of its derived classes (e.g. :class:`OneOffPredictionDataset`,
        :class:`TimeToEventAnalysisDataset` etc.) depending on the type of task.

        See also tutorial ``tutorials/tutorial01_data_format.ipynb`` for examples of use.

        Args:
            time_series (numpy.ndarray | pandas.DataFrame):
                Data representing time series covariates of the samples. Will be initialized as `TimeSeriesSamples`.
            static (numpy.ndarray | pandas.DataFrame, optional):
                Data representing static covariates of the samples. Will be initialized as `StaticSamples`.
                Defaults to `None`.
            targets (numpy.ndarray | pandas.DataFrame, optional):
                Data representing target (outcome) feature(s) of the samples. Will be initialized as
                ``{TimeSeries,Static,Event}Samples`` depending on problem setting in the derived class.
                Defaults to `None`.
            treatments (numpy.ndarray | pandas.DataFrame, optional):
                Data representing treatment (intervention) feature(s) of the samples. Will be initialized as
                ``{TimeSeries,Static,Event}Samples`` depending on problem setting in the derived class.
                Defaults to `None`.
        """
        self.time_series = samples.TimeSeriesSamples(time_series)
        self.static = samples.StaticSamples(static) if static is not None else None

        if targets is not None:
            self._init_predictive(targets=targets, treatments=treatments, **kwargs)
        else:
            self.predictive = None

        self.validate()

    def __rich_repr__(self):
        yield "time_series", RichReprStrPassthrough(self.time_series.short_repr())
        if self.static is not None:
            yield "static", RichReprStrPassthrough(self.static.short_repr())
        if self.predictive is not None:
            yield "static", self.predictive

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    @abc.abstractmethod
    def _init_predictive(
        self,
        targets: data_typing.DataContainer,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:  # pragma: no cover
        """A method to initialize ``self.predictive`` in derived classes."""
        ...

    @property
    def has_static(self) -> bool:
        return self.static is not None

    @property
    def has_predictive_data(self) -> bool:
        return self.predictive is not None

    @property
    def predictive_task(self) -> Union[data_typing.PredictiveTask, None]:
        if self.predictive is not None:
            return self.predictive.predictive_task
        else:
            return None

    def validate(self) -> None:
        """Validate integrity of the dataset."""
        with log_helpers.exc_to_log("Dataset validation failed"):
            if self.static is not None:
                if sorted(self.static.sample_index()) != sorted(self.time_series.sample_index()):
                    raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.static)
            self._validate()

    @abc.abstractmethod
    def _validate(self) -> None:  # pragma: no cover
        ...

    def kfold(self, n_folds: int) -> Generator:
        raise NotImplementedError()


# `Dataset`s corresponding to different tasks follow. More can be added to handle new Tasks.


class OneOffPredictionDataset(Dataset):
    predictive: pred.OneOffPredictionTaskData

    def __init__(  # pylint: disable=useless-super-delegation
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: data_typing.DataContainer,
        static: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`Dataset` subclass for the one-off prediction problem setting, see :class:`Dataset` docs.

        In this setting: ``targets`` are required, will be initialized as `StaticSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: data_typing.DataContainer,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        if targets is None:
            raise ValueError("One-off prediction task requires `targets`")
        self.predictive = pred.OneOffPredictionTaskData(targets=targets, **kwargs)

    def _validate(self) -> None:
        if sorted(self.predictive.targets.sample_index()) != sorted(self.time_series.sample_index()):
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)


class TemporalPredictionDataset(Dataset):
    predictive: pred.TemporalPredictionTaskData

    def __init__(  # pylint: disable=useless-super-delegation
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: data_typing.DataContainer,
        static: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`Dataset` subclass for the temporal prediction problem setting, see :class:`Dataset` docs.

        In this setting: ``targets`` are required, will be initialized as `TimeSeriesSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: data_typing.DataContainer,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        if targets is None:
            raise ValueError("Temporal prediction task requires `targets`")
        self.predictive = pred.TemporalPredictionTaskData(targets=targets, **kwargs)

    def _validate(self) -> None:
        if self.predictive.targets.sample_index() != self.time_series.sample_index():
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)
        # TODO: Possible check - check that .time_series and .predictive.targets have the same time_indexes.


class TimeToEventAnalysisDataset(Dataset):
    predictive: pred.TimeToEventAnalysisTaskData

    def __init__(  # pylint: disable=useless-super-delegation
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: data_typing.DataContainer,
        static: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`Dataset` subclass for the time-to-event analysis problem setting, see :class:`Dataset` docs.

        In this setting: ``targets`` are required, will be initialized as `EventSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(  # pylint: disable=useless-super-delegation
        self,
        targets: data_typing.DataContainer,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        if targets is None:
            raise ValueError("Time-to-event analysis task requires `targets`")
        self.predictive = pred.TimeToEventAnalysisTaskData(targets=targets, **kwargs)

    def _validate(self) -> None:
        if self.predictive.targets.sample_index() != self.time_series.sample_index():
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)
        # TODO: Possible checks - some checks on .time_series and .predictive.targets in terms of
        # their relative position in time?


class OneOffTreatmentEffectsDataset(Dataset):
    predictive: pred.OneOffTreatmentEffectsTaskData

    def __init__(  # pylint: disable=useless-super-delegation
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: data_typing.DataContainer,
        treatments: data_typing.DataContainer,
        static: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`Dataset` subclass for the one-off treatment effects problem setting, see :class:`Dataset` docs.

        In this setting: ``targets`` are required, will be initialized as `TimeSeriesSamples`; ``treatments`` are
        required, will be initialized as `EventSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: data_typing.DataContainer,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        if targets is None:
            raise ValueError("On-off treatment effects task requires `targets`")
        if treatments is None:
            raise ValueError("On-off treatment effects task requires `treatments`")
        self.predictive = pred.OneOffTreatmentEffectsTaskData(targets=targets, treatments=treatments, **kwargs)

    def _validate(self) -> None:
        if self.predictive.targets.sample_index() != self.time_series.sample_index():
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)
        if self.predictive.treatments.sample_index() != self.time_series.sample_index():
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.treatments)
        # TODO: Possible check - check that .time_series and .predictive.targets have the same time_indexes.
        # TODO: Possible checks - some checks on .time_series and .predictive.treatments in terms of
        # their relative position in time?


class TemporalTreatmentEffectsDataset(Dataset):
    predictive: pred.TemporalTreatmentEffectsTaskData

    def __init__(  # pylint: disable=useless-super-delegation
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: data_typing.DataContainer,
        treatments: data_typing.DataContainer,
        static: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`Dataset` subclass for the temporal treatment effects problem setting, see :class:`Dataset` docs.

        In this setting: ``targets`` are required, will be initialized as `TimeSeriesSamples`; ``treatments`` are
        required, will be initialized as `TimeSeriesSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: data_typing.DataContainer,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        if targets is None:
            raise ValueError("Temporal treatment effects task requires `targets`")
        if treatments is None:
            raise ValueError("Temporal treatment effects task requires `treatments`")
        self.predictive = pred.TemporalTreatmentEffectsTaskData(targets=targets, treatments=treatments, **kwargs)

    def _validate(self) -> None:
        if self.predictive.targets.sample_index() != self.time_series.sample_index():
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)
        if self.predictive.treatments.sample_index() != self.time_series.sample_index():
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.treatments)
        # TODO: Possible check - check that .time_series and .predictive.treatments/targets have the same time_indexes.
