"""Data handling for different data samples modalities supported by TemporAI."""

# pylint: disable=unnecessary-ellipsis

import abc
import contextlib
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandera as pa
import pydantic
from packaging.version import Version
from typing_extensions import Self

import tempor.exc
from tempor.core import plugins, pydantic_utils
from tempor.log import log_helpers, logger

from . import data_typing, pandera_utils, utils
from .settings import DATA_SETTINGS


class DataSamples(plugins.Plugin, abc.ABC):
    _data: Any

    @property
    @abc.abstractmethod
    def modality(self) -> data_typing.DataModality:  # pragma: no cover
        """Return the data modality enum corresponding to the class

        Returns:
            data_typing.DataModality: The data modality enum.
        """
        ...

    def __init__(
        self,
        data: data_typing.DataContainer,  # pylint: disable=unused-argument
        **kwargs: Any,
    ) -> None:  # pragma: no cover
        """The abstract base class for all data samples classes.

        Args:
            data (data_typing.DataContainer): The data container.
            **kwargs (Any): Any additional keyword arguments.
        """
        plugins.Plugin.__init__(self)

        if "_skip_validate" not in kwargs:
            # For efficiency, pass `_skip_validate` internally (e.g. in `__getitem__`)
            # when there is no need to validate.
            self.validate()

    def __repr__(self) -> str:
        """The `repr()` representation of the class.

        Returns:
            str: The representation.
        """
        return f"{self.__class__.__name__} with data:\n{self.dataframe()}"

    def _repr_html_(self) -> str:
        """Return a HTML representation of the object, used in Jupyter notebooks.

        Returns:
            str: The HTML representation of the object.
        """
        repr_ = (
            # pylint: disable-next=protected-access
            f'<p><span style="font-family: monospace;">{self.__class__.__name__}</span> with data:</p>'
            f"{self.dataframe()._repr_html_()}"  # pyright: ignore
        )
        return repr_

    def validate(self) -> None:
        """Validate the data contained.

        Raises:
            tempor.exc.DataValidationException: Raised if data validation fails.
        """
        with log_helpers.exc_to_log():
            try:
                self._validate()
            except (
                pa.errors.SchemaError,  # pyright: ignore
                pa.errors.SchemaErrors,  # pyright: ignore
                ValueError,
                TypeError,
            ) as ex:
                raise tempor.exc.DataValidationException(
                    "Data validation failed, see traceback for more details"
                ) from ex

    @abc.abstractmethod
    def _validate(self) -> None:  # pragma: no cover
        """Validate integrity of the data samples. Raise any of `ValueError`, `TypeError`,
        `pandera.errors.SchemaError`, `pandera.errors.SchemaErrors` (or exceptions derived from these) to indicate
        validation failure.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> "DataSamples":  # pragma: no cover
        """Create :class:`DataSamples` from `numpy.ndarray`.

        Args:
            array (np.ndarray):
                The array that represents the data.
            sample_index (Optional[data_typing.SampleIndex], optional):
                List with sample (row) index for each sample. Optional, if `None`, will be of form ``[0, 1, ...]``.
                Defaults to `None`.
            feature_index (Optional[data_typing.FeatureIndex], optional):
                List with feature (column) index for each feature. Optional, if `None`, will be of form
                ``["feat_0", "feat_1", ...]``. Defaults to `None`.
            **kwargs (Any):
                Any additional keyword arguments.

        Returns:
            DataSamples: :class:`DataSamples` object from ``array``.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def from_dataframe(dataframe: pd.DataFrame, **kwargs: Any) -> "DataSamples":  # pragma: no cover
        """Create :class:`DataSamples` from `pandas.DataFrame`."""
        ...

    @abc.abstractmethod
    def numpy(self, **kwargs: Any) -> np.ndarray:  # pragma: no cover
        """Return `numpy.ndarray` representation of the data."""
        ...

    @abc.abstractmethod
    def dataframe(self, **kwargs: Any) -> pd.DataFrame:  # pragma: no cover
        """Return `pandas.DataFrame` representation of the data."""
        ...

    @property
    @abc.abstractmethod
    def num_samples(self) -> int:  # pragma: no cover
        """Return number of samples."""
        ...

    @abc.abstractmethod
    def sample_index(self) -> data_typing.SampleIndex:  # pragma: no cover
        """Return a list representing sample indexes."""
        ...

    def __len__(self) -> int:
        """The length, which is the number of samples.

        Returns:
            int: The number of samples.
        """
        return self.num_samples

    @property
    @abc.abstractmethod
    def num_features(self) -> int:  # pragma: no cover
        """Return number of features."""
        ...

    @abc.abstractmethod
    def short_repr(self) -> str:  # pragma: no cover
        """A short string representation of the object.

        Returns:
            str: The short string representation of the object.
        """
        ...

    @abc.abstractmethod
    def __getitem__(self, key: data_typing.GetItemKey) -> Self:  # pragma: no cover
        """Return a new subset :class:`DataSamples` object with the data indexed by the ``key``.

        Args:
            key (data_typing.GetItemKey): The key to index the data by.

        Returns:
            Self: A new subset :class:`DataSamples` object.
        """
        ...


class StaticSamplesBase(DataSamples):
    @property
    def modality(self) -> data_typing.DataModality:
        """Return the data modality enum corresponding to the class. Here, ``STATIC``.

        Returns:
            data_typing.DataModality: The data modality enum. Here, ``STATIC``.
        """
        return data_typing.DataModality.STATIC


class TimeSeriesSamplesBase(DataSamples):
    @property
    def modality(self) -> data_typing.DataModality:
        """Return the data modality enum corresponding to the class. Here, ``TIME_SERIES``.

        Returns:
            data_typing.DataModality: The data modality enum. Here, ``TIME_SERIES``.
        """
        return data_typing.DataModality.TIME_SERIES

    @abc.abstractmethod
    def time_indexes(self) -> data_typing.TimeIndexList:
        """Get a list containing time indexes for each sample. Each time index is represented as a list of time step
        elements.

        Returns:
            data_typing.TimeIndexList: A list containing time indexes for each sample.
        """
        ...

    @abc.abstractmethod
    def time_indexes_as_dict(self) -> data_typing.SampleToTimeIndexDict:
        """Get a dictionary mapping each sample index to its time index. Time index is represented as a list of time
        step elements.

        Returns:
            data_typing.SampleToTimeIndexDict: The dictionary mapping each sample index to its time index.
        """
        ...

    @abc.abstractmethod
    def time_indexes_float(self) -> List[np.ndarray]:
        """Return time indexes but converting their elements to `float` values.

        Date-time time index will be converted using :obj:`~tempor.data.utils.datetime_time_index_to_float`.

        Returns:
            List[np.ndarray]: List of 1D `numpy.ndarray` s of `float` values, corresponding to the time index.
        """
        ...

    @abc.abstractmethod
    def num_timesteps(self) -> List[int]:
        """Get the number of timesteps for each sample.

        Returns:
            List[int]: List containing the number of timesteps for each sample.
        """
        ...

    @abc.abstractmethod
    def num_timesteps_as_dict(self) -> data_typing.SampleToNumTimestepsDict:
        """Get a dictionary mapping each sample index to its the number of timesteps.

        Returns:
            data_typing.SampleToNumTimestepsDict: List containing the number of timesteps for each sample.
        """
        ...

    @abc.abstractmethod
    def num_timesteps_equal(self) -> bool:
        """Returns `True` if all samples share the same number of timesteps, `False` otherwise.

        Returns:
            bool: whether all samples share the same number of timesteps.
        """
        ...

    @abc.abstractmethod
    def list_of_dataframes(self) -> List[pd.DataFrame]:
        """Returns a list of dataframes where each dataframe has the data for each sample.

        Returns:
            List[pd.DataFrame]: List of dataframes for each sample.
        """
        ...


_DEFAULT_EVENTS_TIME_FEATURE_SUFFIX = "_time"


class EventSamplesBase(DataSamples):
    @property
    def modality(self) -> data_typing.DataModality:
        """Return the data modality enum corresponding to the class. Here, ``EVENT``.

        Returns:
            data_typing.DataModality: The data modality enum. Here, ``EVENT``.
        """
        return data_typing.DataModality.EVENT

    @abc.abstractmethod
    def split(self, time_feature_suffix: str = _DEFAULT_EVENTS_TIME_FEATURE_SUFFIX) -> pd.DataFrame:
        """Return a `pandas.DataFrame` where the time component of each event feature has been split off to its own
        column. The new columns that contain the times will be named ``"<original column name><time_feature_suffix>"``
        and will be inserted before each corresponding ``<original column name>`` column. The ``<original column name>``
        columns will contain only the event value.

        Args:
            time_feature_suffix (str, optional):
                A column name suffix string to identify the time columns that will be split off. Defaults to
                ``"_time"``.

        Returns:
            pd.DataFrame: The output dataframe.
        """
        ...

    @abc.abstractmethod
    def split_as_two_dataframes(
        self, time_feature_suffix: str = _DEFAULT_EVENTS_TIME_FEATURE_SUFFIX
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analogous to :func:`~tempor.data.samples.EventSamples.split` but returns two `pandas.DataFrame` s:
            - first dataframe contains the event times of each feature.
            - second dataframe contains the event values (`True`/`False`) of each feature.

        Args:
            time_feature_suffix (str, optional):
                A column name suffix string to identify the time columns that will be split off. Defaults to
                ``"_time"``.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two `pandas.DataFrame` s containing event times and values respectively.
        """
        ...


def _array_default_sample_index(array: np.ndarray) -> List[int]:
    n_samples, *_ = array.shape
    return list(range(0, n_samples))


def _array_default_feature_index(array: np.ndarray) -> List[str]:
    *_, n_features = array.shape
    return [f"feat_{x}" for x in range(0, n_features)]


def _array_default_time_indexes(array: np.ndarray, padding_indicator: Any) -> List[List[int]]:
    lengths = utils.get_seq_lengths_timeseries_array3d(array, padding_indicator)
    return [list(range(x)) for x in lengths]


plugins.register_plugin_category("static_samples", StaticSamplesBase, plugin_type="dataformat")
plugins.register_plugin_category("time_series_samples", TimeSeriesSamplesBase, plugin_type="dataformat")
plugins.register_plugin_category("event_samples", EventSamplesBase, plugin_type="dataformat")


@plugins.register_plugin(name="static_samples_df", category="static_samples", plugin_type="dataformat")
class StaticSamples(StaticSamplesBase):
    _data: pd.DataFrame
    _schema: pa.DataFrameSchema

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: data_typing.DataContainer,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> None:
        """Create a :class:`StaticSamples` object from the ``data``.

        Args:
            data (data_typing.DataContainer):
                A container with the data.
            sample_index (Optional[data_typing.SampleIndex], optional):
                Used only if ``data`` is a `numpy.ndarray`. List with sample (row) index for each sample. Optional,
                if `None`, will be of form ``[0, 1, ...]``. Defaults to `None`.
            feature_index (Optional[data_typing.FeatureIndex], optional):
                Used only if ``data`` is a `numpy.ndarray`.  List with feature (column) index for each feature.
                Optional, if `None`, will be of form ``["feat_0", "feat_1", ...]``. Defaults to `None`.
            **kwargs (Any):
                Any additional keyword arguments to pass to the constructor.
        """
        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = self._array_to_df(data, sample_index=sample_index, feature_index=feature_index, **kwargs)
        else:  # pragma: no cover  # Prevented by pydantic check.
            raise ValueError(f"Data object {type(data)} not supported")
        super().__init__(data, **kwargs)

    def _validate(self) -> None:
        schema = pandera_utils.init_schema(self._data, coerce=False)
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(schema, pa.DataFrameSchema)  # nosec B101
        logger.debug(f"Inferred schema:\n{schema}")

        # DataFrame-level validation:
        schema = pandera_utils.add_df_checks(
            schema,
            checks_list=[
                pandera_utils.checks.forbid_multiindex_index,
                pandera_utils.checks.forbid_multiindex_columns,
                pandera_utils.checks.configurable.column_index_satisfies_dtype(
                    pandera_utils.UnionDtype[DATA_SETTINGS.feature_index_dtypes],  # type: ignore
                    nullable=DATA_SETTINGS.feature_index_nullable,
                ),
            ],
        )
        self._data = schema.validate(self._data)

        # Values validation:
        schema = pandera_utils.add_regex_column_checks(
            schema,
            regex=".*",
            dtype=pandera_utils.UnionDtype[DATA_SETTINGS.static_value_dtypes],  # type: ignore
            nullable=DATA_SETTINGS.static_values_nullable,
        )
        self._data = schema.validate(self._data)

        # Index validation:
        schema, data = pandera_utils.set_up_index(
            schema,
            self._data,
            dtype=pandera_utils.UnionDtype[DATA_SETTINGS.sample_index_dtypes],  # type: ignore
            name=DATA_SETTINGS.sample_index_name,
            nullable=DATA_SETTINGS.sample_index_nullable,
            coerce=False,
            unique=DATA_SETTINGS.sample_index_unique,
        )
        self._data = schema.validate(data)

        logger.debug(f"Final schema:\n{schema}")
        self._schema = schema

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> "StaticSamples":  # pyright: ignore
        """Create :class:`StaticSamples` from `pandas.DataFrame`. The rows represent samples, the columns represent
        features.

        Args:
            dataframe (pd.DataFrame): The dataframe that represents the data.
            **kwargs (Any): Any additional keyword arguments to pass to the constructor.

        Returns:
            StaticSamples: :class:`StaticSamples` object from ``dataframe``.
        """
        return StaticSamples(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> "StaticSamples":  # pyright: ignore
        """Create :class:`StaticSamples` from `numpy.ndarray`. The 0th dimension represents samples, the 1st dimension
        represents features.

        Args:
            array (np.ndarray): The array with the data.
            sample_index (Optional[data_typing.SampleIndex], optional): Sample indices to assign. Defaults to None.
            feature_index (Optional[data_typing.FeatureIndex], optional): Feature indices to assign. Defaults to None.
            **kwargs (Any): Any additional keyword arguments to pass to the constructor.

        Returns:
            StaticSamples: :class:`StaticSamples` object created from the ``array``.
        """
        return StaticSamples(array, sample_index=sample_index, feature_index=feature_index, **kwargs)

    @staticmethod
    def _array_to_df(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if sample_index is None:
            sample_index = _array_default_sample_index(array)  # pyright: ignore
        if feature_index is None:
            feature_index = _array_default_feature_index(array)
        return pd.DataFrame(data=array, index=sample_index, columns=feature_index, **kwargs)

    def numpy(self, **kwargs: Any) -> np.ndarray:
        """Return the data as a `numpy.ndarray`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            np.ndarray: The `numpy.ndarray`.
        """
        return self._data.to_numpy()

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the data as a `pandas.DataFrame`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            pd.DataFrame: _description_
        """
        return self._data

    def sample_index(self) -> data_typing.SampleIndex:
        """Return a list representing sample indexes.

        Returns:
            data_typing.SampleIndex: Sample indexes.
        """
        return list(self._data.index)  # pyright: ignore

    @property
    def num_samples(self) -> int:
        """Return number of samples.

        Returns:
            int: Number of samples.
        """
        return self._data.shape[0]

    @property
    def num_features(self) -> int:
        """Return number of features.

        Returns:
            int: Number of features.
        """
        return self._data.shape[1]

    def short_repr(self) -> str:
        """A short string representation of the object.

        Returns:
            str: The short representation.
        """
        return f"{self.__class__.__name__}([{self.num_samples}, {self.num_features}])"

    def __getitem__(self, key: data_typing.GetItemKey) -> Self:
        """Return a new subset :class:`StaticSamples` object with the data indexed by the ``key``.

        Args:
            key (data_typing.GetItemKey): The key to index the data by.

        Returns:
            Self: A new subset :class:`StaticSamples` object.
        """
        key_ = utils.ensure_pd_iloc_key_returns_df(key)
        return StaticSamples(  # type: ignore [return-value]
            self._data.iloc[key_, :],  # pyright: ignore
            _skip_validate=True,
        )


@contextlib.contextmanager
def workaround_pandera_pd2_1_0_multiindex_compatibility(schema: pa.DataFrameSchema, data: pd.DataFrame) -> Generator:
    """A version compatibility issue exists between pandera and pandas 2.1.0, as reported here:
    https://github.com/unionai-oss/pandera/issues/1328

    The error pertains to multiindex uniqueness validation giving an unexpected error.

    This is a workaround that will "manually" throw an error that is expected from pandera.
    """

    def problem_versions() -> bool:  # pragma: no cover
        return Version(pd.__version__) >= Version("2.1.0")
        # TODO: When/if fixed in pandera, add the below condition:
        # and Version(pa.__version__) < Version("0.XX.YY")

    try:
        yield

    except ValueError as ex:
        if problem_versions() and (
            "Columns with duplicate values are not supported in stack" in str(ex)
        ):  # pragma: no cover
            cols = data.index.names
            raise pa.errors.SchemaError(  # type: ignore [no-untyped-call]
                schema=schema,
                data=data,
                message=f"columns {cols} not unique",
            )
        else:  # pragma: no cover
            raise

    finally:
        pass


@plugins.register_plugin(name="time_series_samples_df", category="time_series_samples", plugin_type="dataformat")
class TimeSeriesSamples(TimeSeriesSamplesBase):
    _data: pd.DataFrame
    _schema: pa.DataFrameSchema

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: data_typing.DataContainer,
        *,
        padding_indicator: Any = None,
        sample_index: Optional[data_typing.SampleIndex] = None,
        time_indexes: Optional[data_typing.TimeIndexList] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> None:
        """Create a :class:`TimeSeriesSamples` object from the ``data``.

        If ``data`` is a `pandas.DataFrame`, this should be a 2-level multiindex (sample, timestep) dataframe.

        If ``data`` is a `numpy.ndarray`, this should be a 3D array, with dimensions ``(sample, timestep, feature)``.
        Optionally, padding values of ``padding_indicator`` can be set inside the array to pad out the length of arrays
        of different samples in case they differ. Padding needs to go at the end of the timesteps (dim 1). Padding must
        be the same across the feature dimension (dim 2) for each sample.

        Args:
            data (data_typing.DataContainer):
                A container with the data.
            padding_indicator (Any, optional):
                Padding indicator used in ``data`` to indicate padding. Defaults to `None`.
            sample_index (Optional[data_typing.SampleIndex], optional):
                Used only if ``data`` is a `numpy.ndarray`. List with sample (row) index for each sample.
                Optional, if `None`, will be of form ``[0, 1, ...]``. Defaults to `None`.
            time_indexes (Optional[data_typing.TimeIndexList], optional):
                Used only if ``data`` is a `numpy.ndarray`. List of lists containing timesteps for each sample (outer
                list should be the same length as dim 0 of `data`, inner list should contain as many elements as each
                sample has timesteps). Optional, if `None`, will be of form ``[[0, 1, ...], [0, 1, ...], ...]``
                Defaults to `None`.
            feature_index (Optional[data_typing.FeatureIndex], optional):
                Used only if ``data`` is a `numpy.ndarray`.  List with feature (column) index for each feature.
                Optional, if `None`, will be of form ``["feat_0", "feat_1", ...]``.
                Defaults to `None`.
            **kwargs (Any):
                Any additional keyword arguments to pass to the constructor.
        """
        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = self._array_to_df(
                data,
                padding_indicator=padding_indicator,
                sample_index=sample_index,
                time_indexes=time_indexes,
                feature_index=feature_index,
                **kwargs,
            )
        else:  # pragma: no cover  # Prevented by pydantic check.
            raise ValueError(f"Data object {type(data)} not supported")
        super().__init__(data, **kwargs)

    def _validate(self) -> None:
        schema = pandera_utils.init_schema(self._data, coerce=False)
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(schema, pa.DataFrameSchema)  # nosec B101
        logger.debug(f"Inferred schema:\n{schema}")

        # DataFrame-level validation:
        schema = pandera_utils.add_df_checks(
            schema,
            checks_list=[
                pandera_utils.checks.forbid_multiindex_columns,
                pandera_utils.checks.require_2level_multiindex_index,
                pandera_utils.checks.configurable.column_index_satisfies_dtype(
                    pandera_utils.UnionDtype[DATA_SETTINGS.feature_index_dtypes],  # type: ignore
                    nullable=DATA_SETTINGS.feature_index_nullable,
                ),
            ],
        )
        self._data = schema.validate(self._data)

        # Values validation:
        schema = pandera_utils.add_regex_column_checks(
            schema,
            regex=".*",
            dtype=pandera_utils.UnionDtype[DATA_SETTINGS.time_series_value_dtypes],  # type: ignore
            nullable=DATA_SETTINGS.time_series_values_nullable,
        )
        self._data = schema.validate(self._data)

        # Index validation:
        if not (DATA_SETTINGS.sample_index_unique and DATA_SETTINGS.sample_timestep_index_unique):
            raise NotImplementedError("Only supported case: unique sample and unique timestep indexes")
        multiindex_unique_def = (DATA_SETTINGS.sample_index_name, DATA_SETTINGS.time_index_name)
        schema, data = pandera_utils.set_up_2level_multiindex(
            schema,
            self._data,
            dtypes=(
                pandera_utils.UnionDtype[DATA_SETTINGS.sample_index_dtypes],  # type: ignore
                pandera_utils.UnionDtype[DATA_SETTINGS.time_index_dtypes],  # type: ignore
            ),
            names=(DATA_SETTINGS.sample_index_name, DATA_SETTINGS.time_index_name),
            nullable=(DATA_SETTINGS.sample_index_nullable, DATA_SETTINGS.time_index_nullable),
            coerce=False,
            unique=multiindex_unique_def,
        )
        with workaround_pandera_pd2_1_0_multiindex_compatibility(schema, data):
            self._data = schema.validate(data)

        logger.debug(f"Final schema:\n{schema}")
        self._schema = schema

        # TODO:
        # Possible additional validation checks:
        # - Ensure time index sorted ascending within each sample.
        # - Time index float / int expected non-negative values.

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> "TimeSeriesSamples":  # pyright: ignore
        """Create :class:`TimeSeriesSamples` from `pandas.DataFrame`. This row index of the dataframe should be a
        2-level multiindex (sample, timestep). The columns should be the features.

        Args:
            dataframe (pd.DataFrame): The dataframe that contains the data.
            **kwargs (Any): Any additional keyword arguments to pass to the constructor.

        Returns:
            TimeSeriesSamples: The :class:`TimeSeriesSamples` object created from the ``dataframe``.
        """
        return TimeSeriesSamples(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        padding_indicator: Any = None,
        sample_index: Optional[data_typing.SampleIndex] = None,
        time_indexes: Optional[data_typing.TimeIndexList] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> "TimeSeriesSamples":  # pyright: ignore
        """Create :class:`TimeSeriesSamples` from `numpy.ndarray`.

        This should be a 3D array, with dimensions ``(sample, timestep, feature)``.

        Optionally, padding values of ``padding_indicator`` can be set inside the array to pad out the length of arrays
        of different samples in case they differ. Padding needs to go at the end of the timesteps (dim 1). Padding must
        be the same across the feature dimension (dim 2) for each sample.

        Args:
            array (np.ndarray):
                The array that contains the data.
            padding_indicator (Any, optional):
                The padding indicator value. Defaults to `None`.
            sample_index (Optional[data_typing.SampleIndex], optional):
                Sample indexes as a list. Defaults to `None`.
            time_indexes (Optional[data_typing.TimeIndexList], optional):
                Time indexes as a list of list (that is, time indexes per sample). Defaults to `None`.
            feature_index (Optional[data_typing.FeatureIndex], optional):
                Feature indexes as a list. Defaults to `None`.
            **kwargs (Any):
                Any additional keyword arguments.

        Returns:
            TimeSeriesSamples: The :class:`TimeSeriesSamples` object created from the ``array``.
        """
        return TimeSeriesSamples(
            array,
            padding_indicator=padding_indicator,
            sample_index=sample_index,
            time_indexes=time_indexes,
            feature_index=feature_index,
            **kwargs,
        )

    @staticmethod
    def _array_to_df(  # pylint: disable=unused-argument
        array: np.ndarray,
        *,
        padding_indicator: Any,
        sample_index: Optional[data_typing.SampleIndex] = None,
        time_indexes: Optional[data_typing.TimeIndexList] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if sample_index is None:
            sample_index = _array_default_sample_index(array)  # pyright: ignore
        if feature_index is None:
            feature_index = _array_default_feature_index(array)
        if time_indexes is None:
            time_indexes = _array_default_time_indexes(array, padding_indicator)
        if TYPE_CHECKING:  # pragma: no cover
            assert sample_index is not None and feature_index is not None and time_indexes is not None  # nosec B101
        return utils.array3d_to_multiindex_timeseries_dataframe(
            array,
            sample_index=sample_index,
            time_indexes=time_indexes,
            feature_index=feature_index,
            padding_indicator=padding_indicator,
        )

    def numpy(self, *, padding_indicator: Any = DATA_SETTINGS.default_padding_indicator, **kwargs: Any) -> np.ndarray:
        """Return the data as a `numpy.ndarray`.

        Args:
            padding_indicator (Any, optional):
                Padding indicator value. Defaults to `DATA_SETTINGS.default_padding_indicator`.
            **kwargs (Any):
                Any additional keyword arguments. Currently unused.

        Returns:
            np.ndarray: The `numpy.ndarray`.
        """
        return utils.multiindex_timeseries_dataframe_to_array3d(
            df=self._data, padding_indicator=padding_indicator, max_timesteps=None
        )

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the data as a `pandas.DataFrame`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            pd.DataFrame: The `pandas.DataFrame`.
        """
        return self._data

    def sample_index(self) -> data_typing.SampleIndex:
        """Get a list containing sample indexes.

        Returns:
            data_typing.SampleIndex: A list containing sample indexes.
        """
        return list(utils.get_df_index_level0_unique(self._data))  # pyright: ignore

    def time_indexes(self) -> data_typing.TimeIndexList:
        """Get a list containing time indexes for each sample. Each time index is represented as a list of time step
        elements.

        Returns:
            data_typing.TimeIndexList: A list containing time indexes for each sample.
        """
        return list(self.time_indexes_as_dict().values())  # pyright: ignore

    def time_indexes_as_dict(self) -> data_typing.SampleToTimeIndexDict:
        """Get a dictionary mapping each sample index to its time index. Time index is represented as a list of time
        step elements.

        Returns:
            data_typing.SampleToTimeIndexDict: The dictionary mapping each sample index to its time index.
        """
        multiindex = self._data.index
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(multiindex, pd.MultiIndex)  # nosec B101
        sample_index = self.sample_index()
        d = dict()
        for s in sample_index:
            time_index_locs = multiindex.get_locs([s, slice(None)])
            d[s] = list(multiindex.get_level_values(1)[time_index_locs])
        return d  # type: ignore[return-value]

    def time_indexes_float(self) -> List[np.ndarray]:
        """Return time indexes but converting their elements to `float` values.

        Date-time time index will be converted using :obj:`~tempor.data.utils.datetime_time_index_to_float`.

        Returns:
            List[np.ndarray]: List of 1D `numpy.ndarray` s of `float` values, corresponding to the time index.
        """
        return [utils.datetime_time_index_to_float(ti) for ti in self.time_indexes()]

    def num_timesteps(self) -> List[int]:
        """Get the number of timesteps for each sample.

        Returns:
            List[int]: List containing the number of timesteps for each sample.
        """
        return [len(x) for x in self.time_indexes()]

    def num_timesteps_as_dict(self) -> data_typing.SampleToNumTimestepsDict:
        """Get a dictionary mapping each sample index to its the number of timesteps.

        Returns:
            data_typing.SampleToNumTimestepsDict: List containing the number of timesteps for each sample.
        """
        return {key: len(x) for key, x in self.time_indexes_as_dict().items()}  # type: ignore

    def num_timesteps_equal(self) -> bool:
        """Returns `True` if all samples share the same number of timesteps, `False` otherwise.

        Returns:
            bool: whether all samples share the same number of timesteps.
        """
        timesteps = self.num_timesteps()
        return True if len(timesteps) == 0 else all([x == timesteps[0] for x in timesteps])

    def list_of_dataframes(self) -> List[pd.DataFrame]:
        """Returns a list of dataframes where each dataframe has the data for each sample.

        Returns:
            List[pd.DataFrame]: List of dataframes for each sample.
        """
        return utils.multiindex_timeseries_dataframe_to_list_of_dataframes(self._data)

    @property
    def num_samples(self) -> int:
        """Return number of samples.

        Returns:
            int: Number of samples.
        """
        sample_ids = utils.get_df_index_level0_unique(self._data)
        return len(sample_ids)

    @property
    def num_features(self) -> int:
        """Return number of features.

        Returns:
            int: Number of features.
        """
        return self._data.shape[1]

    def short_repr(self) -> str:
        """A short string representation of the object.

        Returns:
            str: The short representation.
        """
        return f"{self.__class__.__name__}([{self.num_samples}, *, {self.num_features}])"

    def __getitem__(self, key: data_typing.GetItemKey) -> Self:
        """Return a subset :class:`TimeSeriesSamples` object with the data indexed by the ``key``.

        Args:
            key (data_typing.GetItemKey): The key to index the data by.

        Returns:
            Self: A new subset :class:`TimeSeriesSamples` object.
        """
        key_ = utils.ensure_pd_iloc_key_returns_df(key)
        sample_index = utils.get_df_index_level0_unique(self._data)
        selected = list(sample_index[key_])  # pyright: ignore
        return TimeSeriesSamples(  # type: ignore [return-value]
            self._data.loc[(selected, slice(None)), :],  # pyright: ignore
            _skip_validate=True,
        )


@plugins.register_plugin(name="event_samples_df", category="event_samples", plugin_type="dataformat")
class EventSamples(EventSamplesBase):
    _data: pd.DataFrame
    _schema: pa.DataFrameSchema
    _schema_split: pa.DataFrameSchema

    @property
    def modality(self) -> data_typing.DataModality:
        """Return the data modality enum corresponding to the class. Here, ``EVENT``.

        Returns:
            data_typing.DataModality: The data modality enum. Here, ``EVENT``.
        """
        return data_typing.DataModality.EVENT

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data: data_typing.DataContainer,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> None:
        """Create an :class:`EventSamples` object from the ``data``.

        Args:
            data (data_typing.DataContainer):
                A container with the data.
            sample_index (Optional[data_typing.SampleIndex], optional):
                Used only if ``data`` is a `numpy.ndarray`. List with sample (row) index for each sample. Optional,
                if `None`, will be of form ``[0, 1, ...]``. Defaults to `None`.
            feature_index (Optional[data_typing.FeatureIndex], optional):
                Used only if ``data`` is a `numpy.ndarray`.  List with feature (column) index for each feature.
                Optional, if `None`, will be of form ``["feat_0", "feat_1", ...]``. Defaults to `None`.
            **kwargs (Any):
                Any additional keyword arguments to pass to the constructor.
        """
        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = self._array_to_df(data, sample_index=sample_index, feature_index=feature_index, **kwargs)
        else:  # pragma: no cover  # Prevented by pydantic check.
            raise ValueError(f"Data object {type(data)} not supported")
        super().__init__(data, **kwargs)

    def _validate(self) -> None:
        schema = pandera_utils.init_schema(self._data, coerce=False)
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(schema, pa.DataFrameSchema)  # nosec B101
        logger.debug(f"Inferred schema:\n{schema}")

        # DataFrame-level validation:
        schema = pandera_utils.add_df_checks(
            schema,
            checks_list=[
                pandera_utils.checks.forbid_multiindex_index,
                pandera_utils.checks.forbid_multiindex_columns,
                pandera_utils.checks.configurable.column_index_satisfies_dtype(
                    pandera_utils.UnionDtype[DATA_SETTINGS.feature_index_dtypes],  # type: ignore
                    nullable=DATA_SETTINGS.feature_index_nullable,
                ),
            ],
        )
        self._data = schema.validate(self._data)

        # Values validation:
        schema = pandera_utils.add_regex_column_checks(
            schema,
            regex=".*",
            dtype=None,
            nullable=DATA_SETTINGS.event_values_nullable,
            checks_list=[pandera_utils.checks.require_element_len_2],
        )
        self._data = schema.validate(self._data)
        # Validate event time and value components:
        suffix = _DEFAULT_EVENTS_TIME_FEATURE_SUFFIX
        data_split = self.split(time_feature_suffix=suffix)
        schema_split = pandera_utils.init_schema(data_split, coerce=False)
        schema_split = pandera_utils.add_regex_column_checks(
            schema_split,
            regex=f".*{suffix}$",  # Event time columns, end in "_time".
            dtype=pandera_utils.UnionDtype[DATA_SETTINGS.time_index_dtypes],  # type: ignore
            nullable=DATA_SETTINGS.time_index_nullable,
        )
        schema_split = pandera_utils.add_regex_column_checks(
            schema_split,
            regex=f"^((?!{suffix}$).)*$",  # Event value columns, do not end in "_time".
            dtype=pandera_utils.UnionDtype[DATA_SETTINGS.event_value_dtypes],  # type: ignore
            nullable=DATA_SETTINGS.event_values_nullable,
        )
        logger.debug(f"Time split-off schema (checks event time and values separately):\n{schema_split}")
        schema_split.validate(data_split)
        self._schema_split = schema_split

        # Index validation:
        schema, data = pandera_utils.set_up_index(
            schema,
            self._data,
            dtype=pandera_utils.UnionDtype[DATA_SETTINGS.sample_index_dtypes],  # type: ignore
            name=DATA_SETTINGS.sample_index_name,
            nullable=DATA_SETTINGS.sample_index_nullable,
            coerce=False,
            unique=DATA_SETTINGS.sample_index_unique,
        )
        self._data = schema.validate(data)

        logger.debug(f"Final schema:\n{schema}")
        self._schema = schema

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> "EventSamples":  # pyright: ignore
        """Create :class:`EventSamples` from `pandas.DataFrame`. The row index of the dataframe should be the sample
        indexes. The columns should be the features. Each feature should contain a tuple of ``(time, value)``
        representing the event.

        Args:
            dataframe (pd.DataFrame): The dataframe that contains the data.
            **kwargs (Any): Any additional keyword arguments to pass to the constructor.

        Returns:
            EventSamples: The :class:`EventSamples` object created from the ``dataframe``.
        """
        return EventSamples(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> "EventSamples":  # pyright: ignore
        """Create :class:`EventSamples` from `numpy.ndarray`. The array should be a 2D array, with dimensions
        ``(sample, feature)``. Each element should contain a tuple of ``(time, value)`` representing the event.

        Args:
            array (np.ndarray): The array that contains the data.
            sample_index (Optional[data_typing.SampleIndex], optional): Sample indexes. Defaults to `None`.
            feature_index (Optional[data_typing.FeatureIndex], optional): Feature index. Defaults to `None`.
            **kwargs (Any): Any additional keyword arguments to pass to the constructor.

        Returns:
            EventSamples: The :class:`EventSamples` object created from the ``array``.
        """
        return EventSamples(array, sample_index=sample_index, feature_index=feature_index, **kwargs)

    @staticmethod
    def _array_to_df(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if sample_index is None:
            sample_index = _array_default_sample_index(array)  # pyright: ignore
        if feature_index is None:
            feature_index = _array_default_feature_index(array)
        return pd.DataFrame(data=array, index=sample_index, columns=feature_index, **kwargs)

    def numpy(self, **kwargs: Any) -> np.ndarray:
        """Return the data as a `numpy.ndarray`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            np.ndarray: The `numpy.ndarray`.
        """
        # TODO: May want at option to return a scikit-survive -style array.
        return self._data.to_numpy()

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the data as a `pandas.DataFrame`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            pd.DataFrame: The `pandas.DataFrame`.
        """
        return self._data

    def sample_index(self) -> data_typing.SampleIndex:
        """Return a list representing sample indexes.

        Returns:
            data_typing.SampleIndex: Sample indexes.
        """
        return list(self._data.index)  # pyright: ignore

    @property
    def num_samples(self) -> int:
        """Return number of samples.

        Returns:
            int: Number of samples.
        """
        return self._data.shape[0]

    @property
    def num_features(self) -> int:
        """Return number of features.

        Returns:
            int: Number of features.
        """
        return self._data.shape[1]

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def split(self, time_feature_suffix: str = _DEFAULT_EVENTS_TIME_FEATURE_SUFFIX) -> pd.DataFrame:
        """Return a `pandas.DataFrame` where the time component of each event feature has been split off to its own
        column. The new columns that contain the times will be named ``"<original column name><time_feature_suffix>"``
        and will be inserted before each corresponding ``<original column name>`` column. The ``<original column name>``
        columns will contain only the event value.

        Args:
            time_feature_suffix (str, optional):
                A column name suffix string to identify the time columns that will be split off. Defaults to
                ``"_time"``.

        Returns:
            pd.DataFrame: The output dataframe.
        """
        df = self._data.copy()
        features = list(df.columns)
        if any(time_feature_suffix in str(c) for c in features):
            raise ValueError(f"Column names must not contain '{time_feature_suffix}'")
        for f_idx, f in enumerate(features):
            df.insert(f_idx * 2, f"{f}{time_feature_suffix}", df[f].apply(lambda x: x[0]))
        for f in features:
            df[f] = df[f].apply(lambda x: x[1])
        return df

    def split_as_two_dataframes(
        self, time_feature_suffix: str = _DEFAULT_EVENTS_TIME_FEATURE_SUFFIX
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analogous to :func:`~tempor.data.samples.EventSamples.split` but returns two `pandas.DataFrame` s:
            - first dataframe contains the event times of each feature.
            - second dataframe contains the event values (`True`/`False`) of each feature.

        Args:
            time_feature_suffix (str, optional):
                A column name suffix string to identify the time columns that will be split off. Defaults to
                ``"_time"``.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two `pandas.DataFrame` s containing event times and values respectively.
        """
        df_split = self.split(time_feature_suffix=time_feature_suffix)
        df_event_times = df_split.loc[:, [c for c in df_split.columns if time_feature_suffix in c]]
        df_event_values = df_split.loc[:, [c for c in df_split.columns if time_feature_suffix not in c]]
        return df_event_times, df_event_values

    def short_repr(self) -> str:
        """A short string representation of the object.

        Returns:
            str: The short representation.
        """
        return f"{self.__class__.__name__}([{self.num_samples}, {self.num_features}])"

    def __getitem__(self, key: data_typing.GetItemKey) -> Self:
        """Return a new subset :class:`EventSamples` object with the data indexed by the ``key``.

        Args:
            key (data_typing.GetItemKey): The key to index the data by.

        Returns:
            Self: A new subset :class:`EventSamples` object.
        """
        key_ = utils.ensure_pd_iloc_key_returns_df(key)
        return EventSamples(  # type: ignore [return-value]
            self._data.iloc[key_, :],  # pyright: ignore
            _skip_validate=True,
        )
