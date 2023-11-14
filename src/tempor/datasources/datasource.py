"""Module defining the data source classes."""

# pylint: disable=unnecessary-ellipsis

import abc
import contextlib
import os
import ssl
from typing import Any, ClassVar, Generator, Optional, Type

import tempor
from tempor.core import plugins
from tempor.data import data_typing, dataset

DATA_DIR = "data"
"""The subdirectory on the user's system where all data source files will be stored.

The full directory will be ``< tempor -> config -> working_directory > / DATA_DIR``.
"""


@contextlib.contextmanager
def monkeypatch_ssl_error_workaround() -> Generator:  # pragma: no cover
    """Some datasets (e.g. UCI diabetes) are hosted on servers that may have SSL issues. This is a workaround that
    monkeypatches the `ssl` module to ignore SSL errors temporarily.
    """

    context_backup = ssl._create_default_https_context  # pylint: disable=protected-access
    ssl._create_default_https_context = ssl._create_unverified_context  # pylint: disable=protected-access

    try:
        yield

    finally:
        # Make sure to restore the original secure context!
        ssl._create_default_https_context = context_backup  # pylint: disable=protected-access


# TODO: Unit test.


class DataSource(plugins.Plugin, abc.ABC):
    """`DataSource` class to ``load`` a `~tempor.data.dataset.DataSet`."""

    data_root_dir: ClassVar[str] = os.path.join(tempor.get_config().get_working_dir(), DATA_DIR)
    """The automatically determined root directory for data on the user's system.
    It will be ``< tempor -> config -> working_directory > / data``.
    """

    def __init__(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        """Initializer for `DataSource`.

        Args:
            **kwargs (Any): Any additional keyword arguments for the :class:`DataSource`.
        """
        plugins.Plugin.__init__(self)

        os.makedirs(self.data_root_dir, exist_ok=True)
        dataset_dir = self.dataset_dir()
        if dataset_dir is not None:
            os.makedirs(dataset_dir, exist_ok=True)

    @staticmethod
    @abc.abstractmethod
    def dataset_dir() -> Optional[str]:  # pragma: no cover
        """The path to the directory where the data file(s) will be stored, if relevant.
        If the data source has no data files, return `None`.

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
        """The expected predictive task of the loaded dataset.

        Returns:
            data_typing.PredictiveTask: Predictive task of loaded dataset.
        """
        ...

    @abc.abstractmethod
    def load(self, **kwargs: Any) -> dataset.PredictiveDataset:  # pragma: no cover
        """The method that should return the loaded dataset for the appropriate ``predictive_task``.

        Args:
            **kwargs (Any): Any additional keyword arguments.

        Returns:
            dataset.PredictiveDataset: The loaded dataset.
        """
        ...


class OneOffPredictionDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """The expected predictive task of the loaded dataset. Here, it is ``ONE_OFF_PREDICTION``.

        Returns:
            data_typing.PredictiveTask: Predictive task of loaded dataset. Here, it is ``ONE_OFF_PREDICTION``.
        """
        return data_typing.PredictiveTask.ONE_OFF_PREDICTION

    @abc.abstractmethod
    def load(self, **kwargs: Any) -> dataset.OneOffPredictionDataset:  # pragma: no cover
        """The method that should return a one-off prediction dataset.

        Args:
            **kwargs (Any): Any additional keyword arguments.

        Returns:
            dataset.OneOffPredictionDataset: The loaded dataset.
        """
        ...


plugins.register_plugin_category("prediction.one_off", OneOffPredictionDataSource, plugin_type="datasource")


class TemporalPredictionDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """The expected predictive task of the loaded dataset. Here, it is ``TEMPORAL_PREDICTION``.

        Returns:
            data_typing.PredictiveTask: Predictive task of loaded dataset. Here, it is ``TEMPORAL_PREDICTION``.
        """
        return data_typing.PredictiveTask.TEMPORAL_PREDICTION

    @abc.abstractmethod
    def load(self, **kwargs: Any) -> dataset.TemporalPredictionDataset:  # pragma: no cover
        """The method that should return a temporal prediction dataset.

        Args:
            **kwargs (Any): Any additional keyword arguments.

        Returns:
            dataset.TemporalPredictionDataset: The loaded dataset.
        """
        ...


plugins.register_plugin_category("prediction.temporal", TemporalPredictionDataSource, plugin_type="datasource")


class TimeToEventAnalysisDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """The expected predictive task of the loaded dataset. Here, it is ``TIME_TO_EVENT_ANALYSIS``.

        Returns:
            data_typing.PredictiveTask: Predictive task of loaded dataset. Here, it is ``TIME_TO_EVENT_ANALYSIS``.
        """
        return data_typing.PredictiveTask.TIME_TO_EVENT_ANALYSIS

    @abc.abstractmethod
    def load(self, **kwargs: Any) -> dataset.TimeToEventAnalysisDataset:  # pragma: no cover
        """The method that should return a time-to-event analysis dataset.

        Args:
            **kwargs (Any): Any additional keyword arguments.

        Returns:
            dataset.TimeToEventAnalysisDataset: The loaded dataset.
        """
        ...


plugins.register_plugin_category("time_to_event", TimeToEventAnalysisDataSource, plugin_type="datasource")


class OneOffTreatmentEffectsDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """The expected predictive task of the loaded dataset. Here, it is ``ONE_OFF_TREATMENT_EFFECTS``.

        Returns:
            data_typing.PredictiveTask: Predictive task of loaded dataset. Here, it is ``ONE_OFF_TREATMENT_EFFECTS``.
        """
        return data_typing.PredictiveTask.ONE_OFF_TREATMENT_EFFECTS

    @abc.abstractmethod
    def load(self, **kwargs: Any) -> dataset.OneOffTreatmentEffectsDataset:  # pragma: no cover
        """The method that should return a one-off treatment effects dataset.

        Args:
            **kwargs (Any): Any additional keyword arguments.

        Returns:
            dataset.OneOffTreatmentEffectsDataset: The loaded dataset.
        """
        ...


plugins.register_plugin_category("treatments.one_off", OneOffTreatmentEffectsDataSource, plugin_type="datasource")


class TemporalTreatmentEffectsDataSource(DataSource):
    @property
    def predictive_task(self) -> data_typing.PredictiveTask:
        """The expected predictive task of the loaded dataset. Here, it is ``TEMPORAL_TREATMENT_EFFECTS``.

        Returns:
            data_typing.PredictiveTask: Predictive task of loaded dataset. Here, it is ``TEMPORAL_TREATMENT_EFFECTS``.
        """
        return data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS

    @abc.abstractmethod
    def load(self, **kwargs: Any) -> dataset.TemporalTreatmentEffectsDataset:  # pragma: no cover
        """The method that should return a temporal treatment effects dataset.

        Args:
            **kwargs (Any): Any additional keyword arguments.

        Returns:
            dataset.TemporalTreatmentEffectsDataset: The loaded dataset.
        """
        ...


plugins.register_plugin_category("treatments.temporal", TemporalTreatmentEffectsDataSource, plugin_type="datasource")
