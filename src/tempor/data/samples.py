"""Data handling for different data samples modalities supported by TemporAI.
"""

# pylint: disable=useless-super-delegation, unnecessary-ellipsis

import abc
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import pandas as pd
import pandera as pa
import pydantic

import tempor.exc
from tempor.log import log_helpers, logger

from . import data_typing, pandera_utils, utils
from .settings import DATA_SETTINGS


class DataSamples(abc.ABC):
    _data: Any

    @property
    @abc.abstractmethod
    def modality(self) -> data_typing.DataModality:  # pragma: no cover
        ...

    def __init__(
        self,
        data: data_typing.DataContainer,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:  # pragma: no cover
        self.validate()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} with data:\n{self.dataframe()}"

    def validate(self) -> None:
        with log_helpers.exc_to_log():
            try:
                self._validate()
            except (
                pa.errors.SchemaError,
                pa.errors.SchemaErrors,
                ValueError,
                TypeError,
            ) as ex:
                raise tempor.exc.DataValidationException(
                    "Data validation failed, see traceback for more details"
                ) from ex

    @abc.abstractmethod
    def _validate(self) -> None:  # pragma: no cover
        """Validate integrity of the data samples. Raise any of `ValueError`, `TypeError`,
        `pandera.errors.SchemaError[s]` (or exceptions derived from these) to indicate validation failure.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> "DataSamples":  # pragma: no cover
        """Create `DataSamples` from `numpy.ndarray`.

        Args:
            array (np.ndarray): The array that represents the data.
            sample_index (List[<sample element>], optional): List with sample (row) index for each sample.
            Optional, if `None`, will be of form `[0, 1, ...]`. Defaults to `None`.
            feature_index (List[<feature element>], optional): List with feature (column) index for each feature.
            Optional, if `None`, will be of form `["feat_0", "feat_1", ...]`. Defaults to `None`.

        Returns:
            DataSamples: `DataSamples` object from `array`.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def from_dataframe(dataframe: pd.DataFrame, **kwargs) -> "DataSamples":  # pragma: no cover
        """
        Create `DataSamples` from `pandas.DataFrame`.
        """
        ...

    @abc.abstractmethod
    def numpy(self, **kwargs) -> np.ndarray:  # pragma: no cover
        """Return `numpy.ndarray` representation of the data."""
        ...

    @abc.abstractmethod
    def dataframe(self, **kwargs) -> pd.DataFrame:  # pragma: no cover
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
        return self.num_samples

    @property
    @abc.abstractmethod
    def num_features(self) -> int:  # pragma: no cover
        """Return number of features."""
        ...

    @abc.abstractmethod
    def short_repr(self) -> str:  # pragma: no cover
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


class StaticSamples(DataSamples):
    _data: pd.DataFrame
    _schema: pa.DataFrameSchema

    @pydantic.validate_arguments(config={"arbitrary_types_allowed": True, "smart_union": True})
    def __init__(
        self,
        data: data_typing.DataContainer,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> None:
        """Create a `StaticSamples` object from the `data`.

        Args:
            data (numpy.ndarray | pandas.DataFrame): A container with the data.
            sample_index (List[<sample element>], optional): Used only if `data` is a `numpy.ndarray`. List with sample
            (row) index for each sample. Optional, if `None`, will be of form `[0, 1, ...]`. Defaults to `None`.
            feature_index (List[<feature element>], optional): Used only if `data` is a `numpy.ndarray`.  List with
            feature (column) index for each feature. Optional, if `None`, will be of form `["feat_0", "feat_1", ...]`.
            Defaults to `None`.
        """
        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = self._array_to_df(data, sample_index=sample_index, feature_index=feature_index, **kwargs)
        else:  # pragma: no cover  # Prevented by pydantic check.
            raise ValueError(f"Data object {type(data)} not supported")
        super().__init__(data, **kwargs)

    @property
    def modality(self) -> data_typing.DataModality:
        return data_typing.DataModality.STATIC

    def _validate(self) -> None:
        schema = pa.infer_schema(self._data)
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(schema, pa.DataFrameSchema)  # nosec B101
        logger.debug(f"Inferred schema:\n{schema}")

        # DataFrame-level validation:
        schema = pandera_utils.add_df_checks(
            schema,
            checks_list=[
                pandera_utils.checks.forbid_multiindex_index,
                pandera_utils.checks.forbid_multiindex_columns,
                pandera_utils.checks.configurable.column_index_satisfies_dtypes(
                    DATA_SETTINGS.feature_index_dtypes, nullable=DATA_SETTINGS.feature_index_nullable
                ),
            ],
        )
        self._data = schema.validate(self._data)

        # Values validation:
        schema = pandera_utils.add_regex_column_checks(
            schema,
            regex=".*",
            dtype=None,
            nullable=DATA_SETTINGS.static_values_nullable,
            checks_list=[pandera_utils.checks.configurable.values_satisfy_dtypes(DATA_SETTINGS.static_value_dtypes)],
        )
        self._data = schema.validate(self._data)

        # Index validation:
        schema, data = pandera_utils.set_up_index(
            schema,
            self._data,
            name=DATA_SETTINGS.sample_index_name,
            nullable=DATA_SETTINGS.sample_index_nullable,
            unique=DATA_SETTINGS.sample_index_unique,
            checks_list=[pandera_utils.checks.configurable.index_satisfies_dtypes(DATA_SETTINGS.sample_index_dtypes)],
        )
        self._data = schema.validate(data)

        logger.debug(f"Final schema:\n{schema}")
        self._schema = schema

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame, **kwargs) -> "StaticSamples":
        return StaticSamples(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> "StaticSamples":
        return StaticSamples(array, sample_index=sample_index, feature_index=feature_index, **kwargs)

    @staticmethod
    def _array_to_df(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if sample_index is None:
            sample_index = _array_default_sample_index(array)  # pyright: ignore
        if feature_index is None:
            feature_index = _array_default_feature_index(array)
        return pd.DataFrame(data=array, index=sample_index, columns=feature_index, **kwargs)

    def numpy(self, **kwargs) -> np.ndarray:
        return self._data.to_numpy()

    def dataframe(self, **kwargs) -> pd.DataFrame:
        return self._data

    def sample_index(self) -> data_typing.SampleIndex:
        return list(self._data.index)  # pyright: ignore

    @property
    def num_samples(self) -> int:
        return self._data.shape[0]

    @property
    def num_features(self) -> int:
        return self._data.shape[1]

    def short_repr(self) -> str:
        return f"{self.__class__.__name__}([{self.num_samples}, {self.num_features}])"


class TimeSeriesSamples(DataSamples):
    _data: pd.DataFrame
    _schema: pa.DataFrameSchema

    @property
    def modality(self) -> data_typing.DataModality:
        return data_typing.DataModality.TIME_SERIES

    @pydantic.validate_arguments(config={"arbitrary_types_allowed": True, "smart_union": True})
    def __init__(
        self,
        data: data_typing.DataContainer,
        *,
        padding_indicator: Any = None,
        sample_index: Optional[data_typing.SampleIndex] = None,
        time_indexes: Optional[data_typing.TimeIndexList] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> None:
        """Create a `TimeSeriesSamples` object from the `data`.

        If `data` is a `pandas.DataFrame`, this should be a 2-level multiindex (sample, timestep) dataframe.

        If `data` is a `numpy.ndarray`, this should be a 3D array, with dimensions `(sample, timestep, feature)`.
        Optionally, padding values of `padding_indicator` can be set inside the array to pad out the length of arrays
        of different samples in case they differ. Padding needs to go at the end of the timesteps (dim 1). Padding must
        be the same across the feature dimension (dim 2) for each sample.

        Args:
            data (numpy.ndarray | pandas.DataFrame): A container with the data.
            padding_indicator (Any, optional): Padding indicator used in `data` to indicate padding. Defaults to None.
            sample_index (List[<sample element>], optional): Used only if `data` is a `numpy.ndarray`. List with sample
            (row) index for each sample. Optional, if `None`, will be of form `[0, 1, ...]`. Defaults to `None`.
            time_indexes (List[List[<timestep element>]], optional): Used only if `data` is a `numpy.ndarray`. List of
            lists containing timesteps for each sample (outer list should be the same length as dim 0 of `data`,
            inner list should contain as many elements as each sample has timesteps). Optional, if `None`, will be of
            form `[[0, 1, ...], [0, 1, ...], ...]` Defaults to None.
            feature_index (List[<feature element>], optional): Used only if `data` is a `numpy.ndarray`.  List with
            feature (column) index for each feature. Optional, if `None`, will be of form `["feat_0", "feat_1", ...]`.
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
        schema = pa.infer_schema(self._data)
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(schema, pa.DataFrameSchema)  # nosec B101
        logger.debug(f"Inferred schema:\n{schema}")

        # DataFrame-level validation:
        schema = pandera_utils.add_df_checks(
            schema,
            checks_list=[
                pandera_utils.checks.forbid_multiindex_columns,
                pandera_utils.checks.require_2level_multiindex_index,
                pandera_utils.checks.configurable.column_index_satisfies_dtypes(
                    DATA_SETTINGS.feature_index_dtypes, nullable=DATA_SETTINGS.feature_index_nullable
                ),
            ],
        )
        self._data = schema.validate(self._data)

        # Values validation:
        schema = pandera_utils.add_regex_column_checks(
            schema,
            regex=".*",
            dtype=None,
            nullable=DATA_SETTINGS.time_series_values_nullable,
            checks_list=[
                pandera_utils.checks.configurable.values_satisfy_dtypes(DATA_SETTINGS.time_series_value_dtypes)
            ],
        )
        self._data = schema.validate(self._data)

        # Index validation:
        if not (DATA_SETTINGS.sample_index_unique and DATA_SETTINGS.sample_timestep_index_unique):
            raise NotImplementedError("Only supported case: unique sample and unique timestep indexes")
        multiindex_unique_def = (DATA_SETTINGS.sample_index_name, DATA_SETTINGS.time_index_name)
        schema, data = pandera_utils.set_up_2level_multiindex(
            schema,
            self._data,
            names=(DATA_SETTINGS.sample_index_name, DATA_SETTINGS.time_index_name),
            nullable=(DATA_SETTINGS.sample_index_nullable, DATA_SETTINGS.time_index_nullable),
            unique=multiindex_unique_def,
            checks_list=(
                [pandera_utils.checks.configurable.index_satisfies_dtypes(DATA_SETTINGS.sample_index_dtypes)],
                [pandera_utils.checks.configurable.index_satisfies_dtypes(DATA_SETTINGS.time_index_dtypes)],
            ),
        )
        self._data = schema.validate(data)

        logger.debug(f"Final schema:\n{schema}")
        self._schema = schema

        # TODO:
        # Possible additional validation checks:
        # - Ensure time index sorted ascending within each sample.
        # - Time index float / int expected non-negative values.

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame, **kwargs) -> "TimeSeriesSamples":
        return TimeSeriesSamples(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        padding_indicator: Any = None,
        sample_index: Optional[data_typing.SampleIndex] = None,
        time_indexes: Optional[data_typing.TimeIndexList] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> "TimeSeriesSamples":
        return TimeSeriesSamples(
            array,
            padding_indicator=padding_indicator,
            sample_index=sample_index,
            time_indexes=time_indexes,
            feature_index=feature_index,
            **kwargs,
        )

    @staticmethod
    def _array_to_df(
        array: np.ndarray,
        *,
        padding_indicator: Any,
        sample_index: Optional[data_typing.SampleIndex] = None,
        time_indexes: Optional[data_typing.TimeIndexList] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> pd.DataFrame:
        if sample_index is None:
            sample_index = _array_default_sample_index(array)  # pyright: ignore
        if feature_index is None:
            feature_index = _array_default_feature_index(array)
        if time_indexes is None:
            time_indexes = _array_default_time_indexes(array, padding_indicator)  # type: ignore
        if TYPE_CHECKING:  # pragma: no cover
            assert sample_index is not None and feature_index is not None and time_indexes is not None  # nosec B101
        return utils.array3d_to_multiindex_timeseries_dataframe(
            array,
            sample_index=sample_index,
            time_indexes=time_indexes,
            feature_index=feature_index,
            padding_indicator=padding_indicator,
        )

    def numpy(self, *, padding_indicator: Any = DATA_SETTINGS.default_padding_indicator, **kwargs) -> np.ndarray:
        return utils.multiindex_timeseries_dataframe_to_array3d(
            df=self._data, padding_indicator=padding_indicator, max_timesteps=None
        )

    def dataframe(self, **kwargs) -> pd.DataFrame:
        return self._data

    def sample_index(self) -> data_typing.SampleIndex:
        return list(self._data.index.levels[0])  # pyright: ignore

    def time_indexes(self) -> data_typing.TimeIndexList:
        """Get a list containing time indexes for each sample. Each time index is represented as a list of time step
        elements.

        Returns:
            List[List[<timestep element>]]: A list containing time indexes for each sample.
        """
        return list(self.time_indexes_as_dict().values())

    def time_indexes_as_dict(self) -> data_typing.SampleToTimeIndexDict:
        """Get a dictionary mapping each sample index to its time index. Time index is represented as a list of time
        step elements.

        Returns:
            Dict[<sample element>, List[<timestep element>]]: A list containing time indexes for each sample.
        """
        multiindex = self._data.index
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(multiindex, pd.MultiIndex)  # nosec B101
        sample_index = list(self._data.index.levels[0])  # pyright: ignore
        d = dict()
        for s in sample_index:
            time_index_locs = multiindex.get_locs(s)
            d[s] = list(multiindex.get_level_values(1)[time_index_locs])
        return d

    # TODO: time indexes sensibly converted to floats would be useful.

    @property
    def num_samples(self) -> int:
        sample_ids = self._data.index.levels[0]  # pyright: ignore
        return len(sample_ids)

    @property
    def num_features(self) -> int:
        return self._data.shape[1]

    def short_repr(self) -> str:
        return f"{self.__class__.__name__}([{self.num_samples}, *, {self.num_features}])"


class EventSamples(DataSamples):
    _data: pd.DataFrame
    _schema: pa.DataFrameSchema
    _schema_split: pa.DataFrameSchema

    @property
    def modality(self) -> data_typing.DataModality:
        return data_typing.DataModality.EVENT

    @pydantic.validate_arguments(config={"arbitrary_types_allowed": True, "smart_union": True})
    def __init__(
        self,
        data: data_typing.DataContainer,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> None:
        """Create an `EventSamples` object from the `data`.

        Args:
            data (numpy.ndarray | pandas.DataFrame): A container with the data.
            sample_index (List[<sample element>], optional): Used only if `data` is a `numpy.ndarray`. List with sample
            (row) index for each sample. Optional, if `None`, will be of form `[0, 1, ...]`. Defaults to `None`.
            feature_index (List[<feature element>], optional): Used only if `data` is a `numpy.ndarray`.  List with
            feature (column) index for each feature. Optional, if `None`, will be of form `["feat_0", "feat_1", ...]`.
            Defaults to `None`.
        """
        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = self._array_to_df(data, sample_index=sample_index, feature_index=feature_index, **kwargs)
        else:  # pragma: no cover  # Prevented by pydantic check.
            raise ValueError(f"Data object {type(data)} not supported")
        super().__init__(data, **kwargs)

    def _validate(self) -> None:
        schema = pa.infer_schema(self._data)
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(schema, pa.DataFrameSchema)  # nosec B101
        logger.debug(f"Inferred schema:\n{schema}")

        # DataFrame-level validation:
        schema = pandera_utils.add_df_checks(
            schema,
            checks_list=[
                pandera_utils.checks.forbid_multiindex_index,
                pandera_utils.checks.forbid_multiindex_columns,
                pandera_utils.checks.configurable.column_index_satisfies_dtypes(
                    DATA_SETTINGS.feature_index_dtypes, nullable=DATA_SETTINGS.feature_index_nullable
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
        suffix = "_time"
        data_split = self.split(time_feature_suffix=suffix)
        schema_split = pa.infer_schema(data_split)
        schema_split = pandera_utils.add_regex_column_checks(
            schema_split,
            regex=f".*{suffix}$",  # Event time columns, end in "_time".
            dtype=None,
            nullable=DATA_SETTINGS.time_index_nullable,
            checks_list=[pandera_utils.checks.configurable.values_satisfy_dtypes(DATA_SETTINGS.time_index_dtypes)],
        )
        schema_split = pandera_utils.add_regex_column_checks(
            schema_split,
            regex=f"^((?!{suffix}$).)*$",  # Event value columns, do not end in "_time".
            dtype=None,
            nullable=DATA_SETTINGS.event_values_nullable,
            checks_list=[pandera_utils.checks.configurable.values_satisfy_dtypes(DATA_SETTINGS.event_value_dtypes)],
        )
        logger.debug(f"Time split-off schema (checks event time and values separately):\n{schema_split}")
        schema_split.validate(data_split)
        self._schema_split = schema_split

        # Index validation:
        schema, data = pandera_utils.set_up_index(
            schema,
            self._data,
            name=DATA_SETTINGS.sample_index_name,
            nullable=DATA_SETTINGS.sample_index_nullable,
            unique=DATA_SETTINGS.sample_index_unique,
            checks_list=[pandera_utils.checks.configurable.index_satisfies_dtypes(DATA_SETTINGS.sample_index_dtypes)],
        )
        self._data = schema.validate(data)

        logger.debug(f"Final schema:\n{schema}")
        self._schema = schema

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame, **kwargs) -> "EventSamples":
        return EventSamples(dataframe, **kwargs)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> "EventSamples":
        return EventSamples(array, sample_index=sample_index, feature_index=feature_index, **kwargs)

    @staticmethod
    def _array_to_df(
        array: np.ndarray,
        *,
        sample_index: Optional[data_typing.SampleIndex] = None,
        feature_index: Optional[data_typing.FeatureIndex] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if sample_index is None:
            sample_index = _array_default_sample_index(array)  # pyright: ignore
        if feature_index is None:
            feature_index = _array_default_feature_index(array)
        return pd.DataFrame(data=array, index=sample_index, columns=feature_index, **kwargs)

    def numpy(self, **kwargs) -> np.ndarray:
        # TODO: May want at option to return a scikit-survive -style array.
        return self._data.to_numpy()

    def dataframe(self, **kwargs) -> pd.DataFrame:
        return self._data

    def sample_index(self) -> data_typing.SampleIndex:
        return list(self._data.index)  # pyright: ignore

    @property
    def num_samples(self) -> int:
        return self._data.shape[0]

    @property
    def num_features(self) -> int:
        return self._data.shape[1]

    @pydantic.validate_arguments(config={"arbitrary_types_allowed": True})
    def split(self, time_feature_suffix: str = "_time") -> pd.DataFrame:
        """Return a `pandas.DataFrame` where the time component of each event feature has been split off to its own
        column. The new columns that contain the times will be named `"<original column name><time_feature_suffix>"`
        and will be inserted before each corresponding `<original column name>` column. The `<original column name>`
        columns will contain only the event value.

        Args:
            time_feature_suffix (str, optional): A column name suffix string to identify the time columns that will be
            split off. Defaults to `"_time"`.

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

    def short_repr(self) -> str:
        return f"{self.__class__.__name__}([{self.num_samples}, {self.num_features}])"
