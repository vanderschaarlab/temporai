"""Module with experimental samples implementations."""

# pyright: reportPrivateImportUsage=false

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from typing_extensions import Self

from tempor.core import plugins
from tempor.log import logger

from . import data_typing, samples, utils
from .settings import DATA_SETTINGS

# NOTE: Dask samples implementations are a work in progress and do not yet fully leverage Dask capabilities.


def _process_npartitions_chunksize(**kwargs: Any) -> Tuple[Optional[int], Optional[int], Any]:
    if "npartitions" in kwargs:
        npartitions = kwargs.pop("npartitions")
        chunksize = None
    elif "chunksize" in kwargs:
        chunksize = kwargs.pop("chunksize")
        npartitions = None
    else:
        npartitions = 1
        chunksize = None
    return npartitions, chunksize, kwargs


@plugins.register_plugin(name="static_samples_dask", category="static_samples", plugin_type="dataformat")
class StaticSamplesDask(samples.StaticSamplesBase):
    _data: dd.DataFrame  # type: ignore[name-defined]
    # _schema: pa.DataFrameSchema

    def __init__(
        self,
        data: data_typing.DataContainer,
        **kwargs: Any,
    ) -> None:
        """Create a :class:`StaticSamplesDask` object from the ``data``.

        Args:
            data (data_typing.DataContainer):
                A container with the data.
            **kwargs (Any):
                Any additional keyword arguments to pass to the constructor.
        """
        npartitions, chunksize, kwargs = _process_npartitions_chunksize(**kwargs)
        if isinstance(data, dd.DataFrame):  # type: ignore[attr-defined]
            self._data = data
        elif isinstance(data, pd.DataFrame):
            self._data = dd.from_pandas(  # type: ignore[attr-defined]
                data,
                npartitions=npartitions,
                chunksize=chunksize,
            )
        elif isinstance(data, np.ndarray):
            raise NotImplementedError("`StaticSamplesDask` does not support `numpy.ndarray` input yet.")
        else:  # pragma: no cover  # Prevented by pydantic check.
            raise ValueError(f"Data object {type(data)} not supported")
        super().__init__(data, **kwargs)

    def _validate(self) -> None:
        # TODO: Validation analogous to the DF implementation.
        logger.info("Validation not yet implemented for Dask data format. Data format consistency is not guaranteed.")

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> "StaticSamplesDask":  # pyright: ignore
        """Create :class:`StaticSamplesDask` from `pandas.DataFrame`. The rows represent samples, the columns represent
        features.

        Args:
            dataframe (pd.DataFrame): The dataframe that represents the data.
            **kwargs (Any): Any additional keyword arguments to pass to the constructor.

        Returns:
            StaticSamplesDask: :class:`StaticSamples` object from ``dataframe``.
        """
        return StaticSamplesDask(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs: Any,
    ) -> "StaticSamplesDask":  # pyright: ignore
        """Not implemented yet."""
        raise NotImplementedError("`StaticSamplesDask` does not support `numpy.ndarray` input yet.")

    def numpy(self, **kwargs: Any) -> np.ndarray:
        """Return the data as a `numpy.ndarray`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            np.ndarray: The `numpy.ndarray`.
        """
        return self._data.compute().to_numpy()

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the data as a `pandas.DataFrame`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            pd.DataFrame: The dataframe.
        """
        return self._data.compute()

    def sample_index(self) -> data_typing.SampleIndex:
        """Return a list representing sample indexes.

        Returns:
            data_typing.SampleIndex: Sample indexes.
        """
        return list(self._data.index.compute())

    @property
    def num_samples(self) -> int:
        """Return number of samples.

        Returns:
            int: Number of samples.
        """
        return self._data.shape[0].compute()

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
        return StaticSamplesDask(  # type: ignore [return-value]
            self._data.compute().iloc[key_, :],  # pyright: ignore
            _skip_validate=True,
        )


def multiindex_df_to_compatible_ddf(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dd.DataFrame:  # type: ignore[name-defined]
    """Convert a multiindex dataframe to a dask dataframe with a single tuple index."""
    compatible_df = pd.DataFrame(data=df.to_numpy(), index=df.index.to_list(), columns=df.columns.to_list())
    return dd.from_pandas(  # type: ignore[attr-defined]
        compatible_df,
        **kwargs,
    )


def compatible_ddf_to_multiindex_df(
    ddf: dd.DataFrame,  # type: ignore[name-defined]
) -> pd.DataFrame:
    """Convert a dask dataframe with a single tuple index to a multiindex dataframe."""
    asdf = ddf.compute()
    df = pd.DataFrame(data=asdf.to_numpy(), columns=asdf.columns.to_list())
    df.index = pd.MultiIndex.from_tuples(asdf.index.to_list())
    return df


@plugins.register_plugin(name="time_series_samples_dask", category="time_series_samples", plugin_type="dataformat")
class TimeSeriesSamplesDask(samples.TimeSeriesSamplesBase):
    _data: dd.DataFrame  # type: ignore[name-defined]
    # _schema: pa.DataFrameSchema

    def __init__(
        self,
        data: data_typing.DataContainer,
        **kwargs: Any,
    ) -> None:
        """Create an :class:`TimeSeriesSamplesDask` object from the ``data``.

        Args:
            data (data_typing.DataContainer):
                A container with the data.
            **kwargs (Any):
                Any additional keyword arguments to pass to the constructor.
        """
        npartitions, chunksize, kwargs = _process_npartitions_chunksize(**kwargs)
        if isinstance(data, dd.DataFrame):  # type: ignore[attr-defined]
            self._data = data
        elif isinstance(data, pd.DataFrame):
            self._data = multiindex_df_to_compatible_ddf(data, npartitions=npartitions, chunksize=chunksize)
        elif isinstance(data, np.ndarray):
            raise NotImplementedError("`TimeSeriesSamples` does not support `numpy.ndarray` input yet.")
        else:  # pragma: no cover  # Prevented by pydantic check.
            raise ValueError(f"Data object {type(data)} not supported")
        super().__init__(data, **kwargs)

    def _validate(self) -> None:
        # TODO: Validation analogous to the DF implementation.
        logger.info("Validation not yet implemented for Dask data format. Data format consistency is not guaranteed.")

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> "TimeSeriesSamplesDask":  # pyright: ignore
        """Create :class:`TimeSeriesSamplesDask` from `pandas.DataFrame`. This row index of the dataframe should be a
        2-level multiindex (sample, timestep). The columns should be the features.

        Args:
            dataframe (pd.DataFrame): The dataframe that contains the data.
            **kwargs (Any): Any additional keyword arguments to pass to the constructor.

        Returns:
            TimeSeriesSamplesDask: The :class:`TimeSeriesSamples` object created from the ``dataframe``.
        """
        return TimeSeriesSamplesDask(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        **kwargs: Any,
    ) -> "TimeSeriesSamplesDask":  # pyright: ignore
        """Not implemented yet."""
        raise NotImplementedError("`TimeSeriesSamples` does not support `numpy.ndarray` input yet.")

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
            df=compatible_ddf_to_multiindex_df(self._data), padding_indicator=padding_indicator, max_timesteps=None
        )

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the data as a `pandas.DataFrame`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            pd.DataFrame: The `pandas.DataFrame`.
        """
        return compatible_ddf_to_multiindex_df(self._data)

    def sample_index(self) -> data_typing.SampleIndex:
        """Get a list containing sample indexes.

        Returns:
            data_typing.SampleIndex: A list containing sample indexes.
        """
        return list(utils.get_df_index_level0_unique(compatible_ddf_to_multiindex_df(self._data)))  # pyright: ignore

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
        multiindex = compatible_ddf_to_multiindex_df(self._data).index
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
        return utils.multiindex_timeseries_dataframe_to_list_of_dataframes(compatible_ddf_to_multiindex_df(self._data))

    @property
    def num_samples(self) -> int:
        """Return number of samples.

        Returns:
            int: Number of samples.
        """
        sample_ids = utils.get_df_index_level0_unique(compatible_ddf_to_multiindex_df(self._data))
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
        sample_index = utils.get_df_index_level0_unique(compatible_ddf_to_multiindex_df(self._data))
        selected = list(sample_index[key_])  # pyright: ignore
        return TimeSeriesSamplesDask(  # type: ignore [return-value]
            compatible_ddf_to_multiindex_df(self._data).loc[(selected, slice(None)), :],  # pyright: ignore
            _skip_validate=True,
        )


@plugins.register_plugin(name="event_samples_dask", category="event_samples", plugin_type="dataformat")
class EventSamplesDask(samples.EventSamplesBase):
    _data: dd.DataFrame  # type: ignore[name-defined]
    # _schema: pa.DataFrameSchema
    # _schema_split: pa.DataFrameSchema

    def __init__(
        self,
        data: data_typing.DataContainer,
        **kwargs: Any,
    ) -> None:
        """Create an :class:`EventSamplesDask` object from the ``data``.

        Args:
            data (data_typing.DataContainer):
                A container with the data.
            **kwargs (Any):
                Any additional keyword arguments to pass to the constructor.
        """
        npartitions, chunksize, kwargs = _process_npartitions_chunksize(**kwargs)
        if isinstance(data, dd.DataFrame):  # type: ignore[attr-defined]
            self._data = data
        elif isinstance(data, pd.DataFrame):
            self._data = dd.from_pandas(  # type: ignore[attr-defined]
                data,
                npartitions=npartitions,
                chunksize=chunksize,
            )
        elif isinstance(data, np.ndarray):
            raise NotImplementedError("`EventSamplesDask` does not support `numpy.ndarray` input yet.")
        else:  # pragma: no cover  # Prevented by pydantic check.
            raise ValueError(f"Data object {type(data)} not supported")
        super().__init__(data, **kwargs)

    def _validate(self) -> None:
        # TODO: Validation analogous to the DF implementation.
        logger.info("Validation not yet implemented for Dask data format. Data format consistency is not guaranteed.")

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> "EventSamplesDask":  # pyright: ignore
        """Create :class:`EventSamples` from `pandas.DataFrame`. The row index of the dataframe should be the sample
        indexes. The columns should be the features. Each feature should contain a tuple of ``(time, value)``
        representing the event.

        Args:
            dataframe (pd.DataFrame): The dataframe that contains the data.
            **kwargs (Any): Any additional keyword arguments to pass to the constructor.

        Returns:
            EventSamplesDask: The :class:`EventSamplesDask` object created from the ``dataframe``.
        """
        return EventSamplesDask(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        **kwargs: Any,
    ) -> "EventSamplesDask":  # pyright: ignore
        """Not implemented yet."""
        raise NotImplementedError("`EventSamplesDask` does not support `numpy.ndarray` input yet.")

    def numpy(self, **kwargs: Any) -> np.ndarray:
        """Return the data as a `numpy.ndarray`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            np.ndarray: The `numpy.ndarray`.
        """
        return self._data.compute().to_numpy()

    def dataframe(self, **kwargs: Any) -> pd.DataFrame:
        """Return the data as a `pandas.DataFrame`.

        Args:
            **kwargs (Any): Any additional keyword arguments. Currently unused.

        Returns:
            pd.DataFrame: The `pandas.DataFrame`.
        """
        return self._data.compute()

    def sample_index(self) -> data_typing.SampleIndex:
        """Return a list representing sample indexes.

        Returns:
            data_typing.SampleIndex: Sample indexes.
        """
        return list(self._data.index.compute())

    @property
    def num_samples(self) -> int:
        """Return number of samples.

        Returns:
            int: Number of samples.
        """
        return self._data.shape[0].compute()

    @property
    def num_features(self) -> int:
        """Return number of features.

        Returns:
            int: Number of features.
        """
        return self._data.shape[1]

    def split(
        self,
        time_feature_suffix: str = samples._DEFAULT_EVENTS_TIME_FEATURE_SUFFIX,  # pylint: disable=protected-access
    ) -> pd.DataFrame:
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
        df = self._data.compute().copy()
        features = list(df.columns)
        if any(time_feature_suffix in str(c) for c in features):  # pragma: no cover
            raise ValueError(f"Column names must not contain '{time_feature_suffix}'")
        for f_idx, f in enumerate(features):
            df.insert(f_idx * 2, f"{f}{time_feature_suffix}", df[f].apply(lambda x: x[0]))
        for f in features:
            df[f] = df[f].apply(lambda x: x[1])
        return df

    def split_as_two_dataframes(
        self,
        time_feature_suffix: str = samples._DEFAULT_EVENTS_TIME_FEATURE_SUFFIX,  # pylint: disable=protected-access
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
        return EventSamplesDask(  # type: ignore [return-value]
            self._data.compute().iloc[key_, :],  # pyright: ignore
            _skip_validate=True,
        )
