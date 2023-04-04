import abc
import os
from typing import ClassVar, Optional, Type  # pylint: disable=unused-import

import tempor

from . import data_typing, dataset

DATA_DIR = "data"
"""The subdirectory on the user's system where all data source files will be stored.

The full directory will be ``< tempor -> config -> working_directory > / DATA_DIR``.
"""


# TODO: Unit test.


class DataLoader(abc.ABC):
    """`DataLoader` class to ``load`` a `~tempor.data.dataset.DataSet`."""

    data_root_dir: ClassVar[str] = os.path.join(tempor.get_config().get_working_dir(), DATA_DIR)
    """The automatically determined root directory for data on the user's system.
    It will be ``< tempor -> config -> working_directory > / data``.
    """

    def __init__(self, **kwargs) -> None:  # pylint: disable=unused-argument
        """Initializer for `DataLoader`.

        Args:
            dataset_dir (Optional[str]):
                Pass in the subdirectory within ``data_root_dir`` where the data source files will be stored,
                if relevant.
            **kwargs:
                Any additional keyword arguments for the :class:`DataLoader`.
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
        ...  # pylint: disable=unnecessary-ellipsis

    @staticmethod
    @abc.abstractmethod
    def url() -> Optional[str]:  # pragma: no cover
        """Define the URL of the data source, if relevant, otherwise return `None`.

        Returns:
            Optional[str]: The URL of the data source, if relevant, otherwise `None`.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    @classmethod
    def requires_internet(cls: "Type[DataLoader]") -> bool:
        """A `classmethod` that returns true if the `DataLoader` requires access to the Internet to `load`
        (at least before it is saved locally).

        Returns `True` if `cls.url()` is not `None`, else returns `False`.

        Returns:
            bool: Whether the `DataLoader` requires Internet access to `load`.
        """
        return cls.url() is not None

    @property
    @abc.abstractmethod
    def predictive_task(self) -> data_typing.PredictiveTask:  # pragma: no cover
        """The expected predictive task of the loaded `~tempor.data.dataset.DataSet`.

        Returns:
            data_typing.PredictiveTask: Predictive task of loaded `~tempor.data.dataset.DataSet`.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.PredictiveDataset:  # pragma: no cover
        """The method that should return the loaded `~tempor.data.dataset.DataSet` for the appropriate
        ``predictive_task``.

        Returns:
            dataset.Dataset: The loaded `~tempor.data.dataset.DataSet`.
        """
        ...  # pylint: disable=unnecessary-ellipsis


class OneOffPredictionDataLoader(DataLoader):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.ONE_OFF_PREDICTION

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.OneOffPredictionDataset:  # pragma: no cover
        ...


class TemporalPredictionDataLoader(DataLoader):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_PREDICTION

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.TemporalPredictionDataset:  # pragma: no cover
        ...


class TimeToEventAnalysisDataLoader(DataLoader):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TIME_TO_EVENT_ANALYSIS

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.TimeToEventAnalysisDataset:  # pragma: no cover
        ...


class OneOffTreatmentEffectsDataLoader(DataLoader):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.ONE_OFF_TREATMENT_EFFECTS

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.OneOffTreatmentEffectsDataset:  # pragma: no cover
        ...


class TemporalTreatmentEffectsDataLoader(DataLoader):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        return data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS

    @abc.abstractmethod
    def load(self, **kwargs) -> dataset.TemporalTreatmentEffectsDataset:  # pragma: no cover
        ...
