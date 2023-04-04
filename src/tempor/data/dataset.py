# pylint: disable=unnecessary-ellipsis

import abc
import dataclasses
from typing import Any, ClassVar, Generator, Optional, Tuple, Union

import rich.pretty
import sklearn.model_selection
from typing_extensions import Self

from tempor.core.utils import RichReprStrPassthrough
from tempor.log import log_helpers, logger

from . import data_typing
from . import predictive as pred
from . import samples, utils

# NOTE: Can probably add other splitters:
Splitter = Union[
    sklearn.model_selection.KFold,
    sklearn.model_selection.StratifiedKFold,
]


@dataclasses.dataclass(frozen=True)
class _SampleIndexMismatchMsg:
    static: ClassVar[str] = (
        "`sample_index` of static samples did not match `sample_index` of time series samples. "
        "Note that the samples need to be in the same order."
    )
    targets: ClassVar[str] = (
        "`sample_index` of targets did not match `sample_index` of time series samples. "
        "Note that the samples need to be in the same order."
    )
    treatments: ClassVar[str] = (
        "`sample_index` of treatments did not match `sample_index` of time series samples. "
        "Note that the samples need to be in the same order."
    )


@dataclasses.dataclass(frozen=True)
class _TimeIndexesMismatchMsg:
    targets: ClassVar[str] = "`time_indexes` of targets did not match `time_indexes` of time series covariates."
    treatments: ClassVar[str] = "`time_indexes` of treatments did not match `time_indexes` of time series covariates."


@dataclasses.dataclass(frozen=True)
class _ExceptionMessages:
    sample_index_mismatch: ClassVar[_SampleIndexMismatchMsg] = _SampleIndexMismatchMsg()
    time_indexes_mismatch: ClassVar[_TimeIndexesMismatchMsg] = _TimeIndexesMismatchMsg()


EXCEPTION_MESSAGES = _ExceptionMessages()
"""Reusable error messages for the module."""


class BaseDataset(abc.ABC):
    _time_series: samples.TimeSeriesSamples
    _static: Optional[samples.StaticSamples]
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
        """Abstract base class representing a dataset used by TemporAI.

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
        self._time_series = samples.TimeSeriesSamples(time_series)
        self._static = samples.StaticSamples(static) if static is not None else None

        self._init_predictive(targets=targets, treatments=treatments, **kwargs)

        self.validate()

    def __rich_repr__(self):
        yield "time_series", RichReprStrPassthrough(self.time_series.short_repr())
        if self.static is not None:
            yield "static", RichReprStrPassthrough(self.static.short_repr())
        if self.predictive is not None:
            yield "predictive", self.predictive

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    @abc.abstractmethod
    def _init_predictive(
        self,
        targets: Optional[data_typing.DataContainer],
        treatments: Optional[data_typing.DataContainer],
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
                if self.static.sample_index() != self.time_series.sample_index():
                    raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.static)
            self._validate()

    @abc.abstractmethod
    def _validate(self) -> None:  # pragma: no cover
        ...

    @property
    def time_series(self) -> samples.TimeSeriesSamples:
        return self._time_series

    @time_series.setter
    def time_series(self, value: samples.TimeSeriesSamples) -> None:
        self._time_series = value
        self.validate()

    @property
    def static(self) -> Optional[samples.StaticSamples]:
        return self._static

    @static.setter
    def static(self, value: Optional[samples.StaticSamples]) -> None:
        self._static = value
        self.validate()

    @property
    @abc.abstractmethod
    def fit_ready(self) -> bool:  # pragma: no cover
        """Returns whether the :class:`BaseDataset` is in a state ready to be ``fit`` on."""
        ...

    def __len__(self) -> int:
        return self.time_series.num_samples

    def __getitem__(self, key: data_typing.GetItemKey) -> Self:
        key_ = utils.ensure_pd_iloc_key_returns_df(key)
        new_dataset = self.__class__(
            time_series=self.time_series[key_].dataframe(),  # pyright: ignore
            static=self.static[key_].dataframe() if self.has_static else None,  # type: ignore[union-attr,index]
            targets=(
                self.predictive.targets[key_].dataframe()  # type: ignore[union-attr]
                if (self.has_predictive_data and self.predictive.targets is not None)  # type: ignore[union-attr]
                else None
            ),
            treatments=(
                self.predictive.treatments[key_].dataframe()  # type: ignore[union-attr]
                if (self.has_predictive_data and self.predictive.treatments is not None)  # type: ignore[union-attr]
                else None
            ),
        )
        return new_dataset

    def train_test_split(
        self,
        *,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    ) -> Tuple[Self, Self]:
        """Split `Dataset` into train and test sets.

        The arguments ``test_size`` ... ``stratify`` are passed to `sklearn.model_selection.train_test_split` to
        generate the split.

        Returns:
            Tuple[Self, Self]: The split tuple ``(dataset_train, dataset_test)``.
        """
        sample_ilocs = list(range(len(self)))
        sample_ilocs_train, sample_ilocs_test = sklearn.model_selection.train_test_split(
            sample_ilocs,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        return self[sample_ilocs_train], self[sample_ilocs_test]

    def split(
        self,
        splitter: Splitter,
        **kwargs,
    ) -> Generator[Tuple[Self, Self], None, None]:
        """Generate dataset splits according to the scikit-learn ``splitter`` (`~tempor.data.dataset.Splitter`).
        The ``kwargs`` are passed to the underlying splitter's ``split`` method.

        Example:
            >>> from sklearn.model_selection import KFold
            >>> from tempor.utils.dataloaders import SineDataLoader
            >>> data = SineDataLoader().load()
            >>> kfold = KFold(n_splits=5)
            >>> len([(data_train, data_test) for (data_train, data_test) in data.split(splitter=kfold)])
            5

        Args:
            splitter (Splitter): A `sklearn` splitter.

        Yields:
            Generator[Tuple[Self, Self], None, None]: ``(dataset_train, dataset_test)`` for each split.
        """
        sample_ilocs: Any = list(range(len(self)))
        for sample_ilocs_train, sample_ilocs_test in splitter.split(X=sample_ilocs, **kwargs):
            yield self[sample_ilocs_train], self[sample_ilocs_test]


# `Dataset`s corresponding to different tasks follow. More can be added to handle new Tasks.

# TODO: unit test CovariatesDataset.
class CovariatesDataset(BaseDataset):
    def __init__(
        self,
        time_series: data_typing.DataContainer,
        *,
        static: Optional[data_typing.DataContainer] = None,
        targets: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`BaseDataset` subclass for a dataset that does not contain any predictive data
        (``targets`` or ``treatments``).
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: Optional[data_typing.DataContainer],
        treatments: Optional[data_typing.DataContainer],
        **kwargs,
    ) -> None:
        if targets is not None:
            raise ValueError(f"`targets` must not be set for a {self.__class__.__name__}.")
        if treatments is not None:
            raise ValueError(f"`treatments` must not be set for a {self.__class__.__name__}.")
        self.predictive = None

    @property
    def fit_ready(self) -> bool:
        return True


class PredictiveDataset(BaseDataset):
    def __init__(
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: Optional[data_typing.DataContainer],
        static: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`BaseDataset` subclass for a dataset that can contain predictive data
        (``targets`` or ``treatments``).

        This is an abstract class, to be derived from for different predictive task -specific ``Dataset`` s.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    @property
    @abc.abstractmethod
    def predict_ready(self) -> bool:  # pragma: no cover
        """Returns whether the :class:`PredictiveDataset` is in a state ready to be ``predict``ed on."""
        ...


class OneOffPredictionDataset(PredictiveDataset):
    predictive: pred.OneOffPredictionTaskData

    def __init__(
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: Optional[data_typing.DataContainer],
        static: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`PredictiveDataset` subclass for the one-off prediction problem setting,
        see :class:`BaseDataset` docs.

        In this setting: ``targets`` are required for fitting, will be initialized as `StaticSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: Optional[data_typing.DataContainer],
        treatments: Optional[data_typing.DataContainer],
        **kwargs,
    ) -> None:
        if targets is None:
            logger.debug(
                f"`targets` provided was None for {self.__class__.__name__}, "
                "this Dataset can only be used for prediction not fitting"
            )
        self.predictive = pred.OneOffPredictionTaskData(parent_dataset=self, targets=targets, **kwargs)

    def _validate(self) -> None:
        if self.predictive.targets is not None:
            if self.predictive.targets.sample_index() != self.time_series.sample_index():
                raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)

    @property
    def fit_ready(self) -> bool:
        return self.predictive.targets is not None

    @property
    def predict_ready(self) -> bool:
        return True


class TemporalPredictionDataset(PredictiveDataset):
    predictive: pred.TemporalPredictionTaskData

    def __init__(
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: Optional[data_typing.DataContainer],
        static: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`PredictiveDataset` subclass for the temporal prediction problem setting,
        see :class:`BaseDataset` docs.

        In this setting: ``targets`` are required for fitting, will be initialized as `TimeSeriesSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: Optional[data_typing.DataContainer],
        treatments: Optional[data_typing.DataContainer],
        **kwargs,
    ) -> None:
        if targets is None:
            logger.debug(
                f"`targets` provided was None for {self.__class__.__name__}, "
                "this Dataset can only be used for prediction not fitting"
            )
        self.predictive = pred.TemporalPredictionTaskData(parent_dataset=self, targets=targets, **kwargs)

    def _validate(self) -> None:
        if self.predictive.targets is not None:
            if self.predictive.targets.sample_index() != self.time_series.sample_index():
                raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)
            if self.predictive.targets.time_indexes() != self.time_series.time_indexes():
                raise ValueError(EXCEPTION_MESSAGES.time_indexes_mismatch.targets)

    @property
    def fit_ready(self) -> bool:
        return self.predictive.targets is not None

    @property
    def predict_ready(self) -> bool:
        return True


class TimeToEventAnalysisDataset(PredictiveDataset):
    predictive: pred.TimeToEventAnalysisTaskData

    def __init__(
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: Optional[data_typing.DataContainer],
        static: Optional[data_typing.DataContainer] = None,
        treatments: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`PredictiveDataset` subclass for the time-to-event analysis problem setting,
        see :class:`BaseDataset` docs.

        In this setting: ``targets`` are required for fitting, will be initialized as `EventSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: Optional[data_typing.DataContainer],
        treatments: Optional[data_typing.DataContainer],
        **kwargs,
    ) -> None:
        if targets is None:
            logger.debug(
                f"`targets` provided was None for {self.__class__.__name__}, "
                "this Dataset can only be used for prediction not fitting"
            )
        self.predictive = pred.TimeToEventAnalysisTaskData(parent_dataset=self, targets=targets, **kwargs)

    def _validate(self) -> None:
        if self.predictive.targets is not None:
            if self.predictive.targets.sample_index() != self.time_series.sample_index():
                raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)
        # TODO: Possible checks - some checks on .time_series and .predictive.targets in terms of
        # their relative position in time?

    @property
    def fit_ready(self) -> bool:
        return self.predictive.targets is not None

    @property
    def predict_ready(self) -> bool:
        return True


class OneOffTreatmentEffectsDataset(PredictiveDataset):
    predictive: pred.OneOffTreatmentEffectsTaskData

    def __init__(
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: Optional[data_typing.DataContainer],
        treatments: data_typing.DataContainer,
        static: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`PredictiveDataset` subclass for the one-off treatment effects problem setting,
        see :class:`BaseDataset` docs.

        In this setting: ``targets`` are required for fitting, will be initialized as `TimeSeriesSamples`;
        ``treatments`` are required for both fitting and prediction, will be initialized as `EventSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: Optional[data_typing.DataContainer],
        treatments: Optional[data_typing.DataContainer],
        **kwargs,
    ) -> None:
        if targets is None:
            logger.debug(
                f"`targets` provided was None for {self.__class__.__name__}, "
                "this Dataset can only be used for prediction not fitting"
            )
        if treatments is None:
            raise ValueError("One-off treatment effects task requires `treatments`")
        self.predictive = pred.OneOffTreatmentEffectsTaskData(
            parent_dataset=self, targets=targets, treatments=treatments, **kwargs
        )

    def _validate(self) -> None:
        if self.predictive.targets is not None:
            if self.predictive.targets.sample_index() != self.time_series.sample_index():
                raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)
            if self.predictive.targets.time_indexes() != self.time_series.time_indexes():
                raise ValueError(EXCEPTION_MESSAGES.time_indexes_mismatch.targets)
        if self.predictive.treatments.sample_index() != self.time_series.sample_index():
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.treatments)
        # TODO: Possible checks - some checks on .time_series and .predictive.treatments in terms of
        # their relative position in time?

    @property
    def fit_ready(self) -> bool:
        return self.predictive.targets is not None and self.predictive.treatments is not None

    @property
    def predict_ready(self) -> bool:
        return self.predictive.treatments is not None


class TemporalTreatmentEffectsDataset(PredictiveDataset):
    predictive: pred.TemporalTreatmentEffectsTaskData

    def __init__(
        self,
        time_series: data_typing.DataContainer,
        *,
        targets: Optional[data_typing.DataContainer],
        treatments: data_typing.DataContainer,
        static: Optional[data_typing.DataContainer] = None,
        **kwargs,
    ) -> None:
        """A :class:`PredictiveDataset` subclass for the temporal treatment effects problem setting,
        see :class:`BaseDataset` docs.

        In this setting: ``targets`` are required for fitting, will be initialized as `TimeSeriesSamples`;
        ``treatments`` are required for both fitting and prediction, will be initialized as `TimeSeriesSamples`.
        """
        super().__init__(time_series=time_series, static=static, targets=targets, treatments=treatments, **kwargs)

    def _init_predictive(
        self,
        targets: Optional[data_typing.DataContainer],
        treatments: Optional[data_typing.DataContainer],
        **kwargs,
    ) -> None:
        if targets is None:
            logger.debug(
                f"`targets` provided was None for {self.__class__.__name__}, "
                "this Dataset can only be used for prediction not fitting"
            )
        if treatments is None:
            raise ValueError("Temporal treatment effects task requires `treatments`")
        self.predictive = pred.TemporalTreatmentEffectsTaskData(
            parent_dataset=self, targets=targets, treatments=treatments, **kwargs
        )

    def _validate(self) -> None:
        if self.predictive.targets is not None:
            if self.predictive.targets.sample_index() != self.time_series.sample_index():
                raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.targets)
            if self.predictive.targets.time_indexes() != self.time_series.time_indexes():
                raise ValueError(EXCEPTION_MESSAGES.time_indexes_mismatch.targets)
        if self.predictive.treatments.sample_index() != self.time_series.sample_index():
            raise ValueError(EXCEPTION_MESSAGES.sample_index_mismatch.treatments)
        if self.predictive.treatments.time_indexes() != self.time_series.time_indexes():
            raise ValueError(EXCEPTION_MESSAGES.time_indexes_mismatch.treatments)

    @property
    def fit_ready(self) -> bool:
        return self.predictive.targets is not None and self.predictive.treatments is not None

    @property
    def predict_ready(self) -> bool:
        return self.predictive.treatments is not None
