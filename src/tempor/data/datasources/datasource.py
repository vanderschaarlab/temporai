# pylint: disable=unnecessary-ellipsis

import abc
import os
from typing import ClassVar, Optional, Type

import tempor

from .. import data_typing, dataset

DATA_DIR = "data"
"""The subdirectory on the user's system where all data source files will be stored.

The full directory will be ``< tempor -> config -> working_directory > / DATA_DIR``.
"""


# TODO: Unit test.


class DataSource(abc.ABC):
    """`DataSource` class to ``load`` a `~tempor.data.dataset.DataSet`."""

    data_root_dir: ClassVar[str] = os.path.join(tempor.get_config().get_working_dir(), DATA_DIR)
    """The automatically determined root directory for data on the user's system.
    It will be ``< tempor -> config -> working_directory > / data``.
    """

    def __init__(self, **kwargs) -> None:  # pylint: disable=unused-argument
        """Initializer for `DataSource`.

        Args:
            dataset_dir (Optional[str]):
                Pass in the subdirectory within ``data_root_dir`` where the data source files will be stored,
                if relevant.
            **kwargs:
                Any additional keyword arguments for the :class:`DataSource`.
        """
        os.makedirs(self.data_root_dir, exist_ok=True)
        dataset_dir = self.dataset_dir()
        if dataset_dir is not None:
            os.makedirs(dataset_dir, exist_ok=True)

    @staticmethod
    @abc.abstractmethod
    def dataset_dir() -> Optional[str]:  # pragma: no cover
        """The path to the directory where the data file(s) will be stored, if relevant.
        If the data loader has no data files, return `None`

        Note:
            the path should correspond to a subdirectory within ``data_root_dir``.

        Returns:
            Optional[str]: The path of the directory where the data file(s) will be stored, if relevant, else `None`.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def url() -> Optional[str]:  # pragma: no cover
        """Define the URL of the data source, if relevant, otherwise return `None`.

        Returns:
            Optional[str]: The URL of the data source, if relevant, otherwise `None`.
        """
        ...

    @classmethod
    def requires_internet(cls: "Type[DataSource]") -> bool:
        """A `classmethod` that returns true if the `DataSource` requires access to the Internet to `load`
        (at least before it is saved locally).

        Returns `True` if `cls.url()` is not `None`, else returns `False`.

        Returns:
            bool: Whether the `DataSource` requires Internet access to `load`.
        """
        return cls.url() is not None

    @property
    @abc.abstractmethod
    def predictive_task(self) -> data_typing.PredictiveTask:  # pragma: no cover
        """The expected predictive task of the loaded `~tempor.data.dataset.DataSet`.

        Returns:
            data_typing.PredictiveTask: Predictive task of loaded `~tempor.data.dataset.DataSet`.
        """
        ...

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.PredictiveDataset:  # pragma: no cover
        """The method that should return the loaded `~tempor.data.dataset.DataSet` for the appropriate
        ``predictive_task``.

        Returns:
            dataset.Dataset: The loaded `~tempor.data.dataset.DataSet`.
        """
        ...


class OneOffPredictionDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.ONE_OFF_PREDICTION

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.OneOffPredictionDataset:  # pragma: no cover
        ...


class TemporalPredictionDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_PREDICTION

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.TemporalPredictionDataset:  # pragma: no cover
        ...


class TimeToEventAnalysisDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TIME_TO_EVENT_ANALYSIS

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.TimeToEventAnalysisDataset:  # pragma: no cover
        ...


class OneOffTreatmentEffectsDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.ONE_OFF_TREATMENT_EFFECTS

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.OneOffTreatmentEffectsDataset:  # pragma: no cover
        ...


class TemporalTreatmentEffectsDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.TemporalTreatmentEffectsDataset:  # pragma: no cover
        ...
