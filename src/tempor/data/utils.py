import dataclasses
import itertools
from typing import Any, ClassVar, List, Optional

import numpy as np
import pandas as pd
import pydantic

from . import data_typing


@dataclasses.dataclass(frozen=True)
class _ExceptionMessages:
    expected_array1d: ClassVar[str] = "Expected 1d array"
    expected_array2d: ClassVar[str] = "Expected 2d array"
    expected_array3d: ClassVar[str] = "Expected 3d array"
    expected_array_dtype_bool: ClassVar[str] = "Expected array of dtype bool"
    padding_indicator_nan_not_supported: ClassVar[str] = "Padding indicator of `numpy.nan` is not supported"


EXCEPTION_MESSAGES = _ExceptionMessages()
"""Reusable error messages for the module."""


# --- Multiindex timeseries dataframe --> 3D numpy array. ---


def value_in_df(df: pd.DataFrame, *, value: Any) -> bool:
    """Check if `value` exists in dataframe `df`, accounting for the case where `value` is `nan`."""
    return (pd.isnull(value) and df.isna().any().any()) or (df == value).any().any()


@pydantic.validate_arguments(config={"arbitrary_types_allowed": True, "smart_union": True})
def multiindex_timeseries_dataframe_to_array3d(
    df: pd.DataFrame, *, padding_indicator: Any, max_timesteps: Optional[int] = None
) -> np.ndarray:
    """Convert timeseries dataframe `df` with a 2-level multiindex (sample, timestep) to a 3D numpy array with
    dimensions `(sample, timestep, feature)`.

    Args:
        df (pd.DataFrame): Input dataframe
        padding_indicator (Any): padding indicator value to use to pad the output array in case of unequal number of
        timesteps for different samples.
        max_timesteps (int, optional): Maximum number of timesteps to use. This will become the size of the dim 1 of the
        output array. If set to `None`, this dimension will be set as the highest number of timesteps among the samples.
        Defaults to None.

    Raises:
        ValueError: raised if the `padding_indicator` found as one of the data values in `df`.

    Returns:
        np.ndarray: Output 3D numpy array.
    """
    if value_in_df(df, value=padding_indicator):
        raise ValueError(f"Value `{padding_indicator}` found in data frame, choose a different padding indicator")
    samples = df.index.get_level_values(level=0).unique()
    num_samples = len(samples)
    num_features = len(df.columns)
    num_timesteps_per_sample = df.groupby(level=0).size()
    max_actual_timesteps = num_timesteps_per_sample.max()
    max_timesteps = max_actual_timesteps if max_timesteps is None else max_timesteps
    array = np.full(shape=(num_samples, max_timesteps, num_features), fill_value=padding_indicator)
    for i_sample, idx_sample in enumerate(samples):
        set_vals = df.loc[idx_sample, :, :].to_numpy()[:max_timesteps, :]  # pyright: ignore
        array[i_sample, : num_timesteps_per_sample[idx_sample], :] = set_vals  # pyright: ignore
    return array


# --- 3D numpy array --> Multiindex timeseries dataframe. ---


def check_bool_array1d_trues_consecutive(array: np.ndarray, at_beginning: bool = False, at_end: bool = False) -> bool:
    """Check if 1D `array` (containing `bool` values) has all `True` elements consecutively. If `at_{beginning,end}` is
    set, will also check that a `True` element is present as the first or last element of the `array`, respectively.
    Raises `ValueError` if input `array` format is unexpected.

    Examples:
    >>> import numpy as np
    >>> check_bool_array1d_trues_consecutive(np.asarray([False, True, True, True, False]))
    True
    >>> check_bool_array1d_trues_consecutive(np.asarray([False, True, False, True, False]))
    False
    >>> check_bool_array1d_trues_consecutive(np.asarray([False, True, True, True]), at_end=True)
    True
    """
    if array.ndim != 1:
        raise ValueError(EXCEPTION_MESSAGES.expected_array1d)
    if array.dtype != bool:
        raise ValueError(EXCEPTION_MESSAGES.expected_array_dtype_bool)
    if len(array) == 0:
        return True
    true_idxs = np.where(array)[0]
    if len(true_idxs) == 0:
        return True
    if not (np.ediff1d(true_idxs) == 1).all():
        return False
    else:
        check_passed = True
        if at_beginning:
            check_passed = check_passed and 0 in true_idxs
        if at_end:
            check_passed = check_passed and (len(array) - 1) in true_idxs
        return check_passed


def check_bool_array2d_identical_along_dim1(array: np.ndarray) -> bool:
    """Check if 2D `array` (containing `bool` values) has the same values along dimension 1.

    Examples:
    >>> import numpy as np
    >>> check_bool_array2d_identical_along_dim1(np.asarray([[True, True, False], [True, True, False]]).T)
    True
    >>> check_bool_array2d_identical_along_dim1(np.asarray([[True, True, False], [False, True, False]]).T)
    False
    """
    if array.ndim != 2:
        raise ValueError(EXCEPTION_MESSAGES.expected_array2d)
    if array.dtype != bool:
        raise ValueError(EXCEPTION_MESSAGES.expected_array_dtype_bool)
    return (np.diff(array.astype(int), axis=1) == 0).all()


def get_array1d_length_until_padding(array: np.ndarray, padding_indicator: Any = None) -> int:
    """Get the length of 1D `array` up to first padding indicated by `padding_indicator`. Raises `ValueError` if input
    `array` format is unexpected.

    Examples:
    >>> import numpy as np
    >>> pad = 999.0
    >>> get_array1d_length_until_padding(np.asarray([1, 8, -3, 9, pad]), padding_indicator=pad)
    4
    >>> get_array1d_length_until_padding(np.asarray([1, 8, -3, 9, 5]), padding_indicator=pad)
    5
    """
    if array.ndim != 1:
        raise ValueError(EXCEPTION_MESSAGES.expected_array1d)
    if np.isnan(padding_indicator):
        raise ValueError(EXCEPTION_MESSAGES.padding_indicator_nan_not_supported)
    array_padding = array == padding_indicator
    positions = np.where(array_padding)[0]
    if len(positions) != 0:
        return positions[0]
    else:
        return len(array)


def validate_timeseries_array3d(array: np.ndarray, padding_indicator: Any = None):
    """Check if 3D `array` representing timeseries satisfies the blow criteria, otherwise raise ValueError:
    - 3 dimensions,
    - Dimension 2 not of size 0,
    - If `padding_indicator` is provided, also check it is not `np.nan`, as this is not supported.
    """
    if array.ndim != 3:
        raise ValueError(EXCEPTION_MESSAGES.expected_array3d)
    if array.shape[2] == 0:
        raise ValueError("Dim 2 (-1) is the feature dimension and must not be of size 0")
    if padding_indicator is not None and np.isnan(padding_indicator):
        raise ValueError(EXCEPTION_MESSAGES.padding_indicator_nan_not_supported)


def get_seq_lengths_timeseries_array3d(array: np.ndarray, padding_indicator: Any = None) -> List[int]:
    """Given a 3D numpy `array` that represents timeseries like `(sample, timestep, feature)`, and optionally a
    `padding_indicator` to indicate padding, get the length (number of [non-padding] timesteps) for each sample.

    Example:
    >>> import numpy as np
    >>> pad = 999.0
    >>> array = np.asarray(  # Array with two samples, with two timeseries features.
    ...     [
    ...         # Sample 1:
    ...         [
    ...             [11, 12, 13, 14, pad],
    ...             [1.1, 1.2, 1.3, 1.4, pad],
    ...         ],
    ...         # Sample 2:
    ...         [
    ...             [21, 22, pad, pad, pad],
    ...             [2.1, 2.2, pad, pad, pad],
    ...         ],
    ...     ]
    ... )
    >>> array = np.transpose(array, (0, 2, 1))
    >>> get_seq_lengths_timeseries_array3d(array, padding_indicator=pad)
    [4, 2]
    """
    validate_timeseries_array3d(array, padding_indicator)
    is_padded = padding_indicator is not None
    if is_padded:
        lengths = []
        for array_sample in array:
            array_sample_padding = array_sample == padding_indicator
            if not check_bool_array2d_identical_along_dim1(array_sample_padding):
                raise ValueError(
                    "Expected padding to be indicated identically across all features for each sample. "
                    f"Problem sample as array:\n{array_sample}"
                )
            array_sample_feat0, array_sample_padding_feat0 = array_sample[:, 0], array_sample_padding[:, 0]
            if not check_bool_array1d_trues_consecutive(array_sample_padding_feat0, at_end=True):
                raise ValueError(
                    "Expected all padding values to be consecutive and at the end. "
                    f"Problem sample 0th feature as array:\n{array_sample[0, :]}"
                )
            n_timesteps = get_array1d_length_until_padding(array_sample_feat0, padding_indicator)
            lengths.append(n_timesteps)
        return lengths
    else:
        return [array.shape[1]] * array.shape[0]


def unpad_timeseries_array3d(array: np.ndarray, padding_indicator: Any) -> List[np.ndarray]:
    """Given a 3D numpy `array` that represents timeseries like `(sample, timestep, feature)`, and optionally a
    `padding_indicator` to indicate padding, return a list of length `num_samples`, which contains arrays for each sample
    like `(timestep, feature)`, with the padding removed.
    """
    validate_timeseries_array3d(array, padding_indicator)
    lengths = get_seq_lengths_timeseries_array3d(array, padding_indicator)
    arrays_unpadded = []
    for sample_i, length in enumerate(lengths):
        arrays_unpadded.append(array[sample_i, :length, :])
    return arrays_unpadded


def make_sample_time_index_tuples(
    sample_index: data_typing.SampleIndex, time_indexes: data_typing.TimeIndexList
) -> data_typing.SampleTimeIndexTuples:
    """Given a list of elements `sample_index` representing sample IDs and a list (of same length) of lists each
    representing the timesteps for the corresponding sample, return a list of tuples like
    `[(<sample ID>, <timestep>), ...]`.

    Example:
    >>> sample_index = ["s1", "s2"]
    >>> time_indexes = [[1, 2, 3], [1, 5, 9, 10]]
    >>> make_sample_time_index_tuples(sample_index, time_indexes)
    [('s1', 1), ('s1', 2), ('s1', 3), ('s2', 1), ('s2', 5), ('s2', 9), ('s2', 10)]
    """
    if len(sample_index) != len(time_indexes):
        raise ValueError("Expected the same number of elements in `sample_index` and `time_indexes`")
    sample_indexes_copied = [[si] * len(tis) for si, tis in zip(sample_index, time_indexes)]
    sample_indexes_flattened = list(itertools.chain.from_iterable(sample_indexes_copied))
    time_indexes_flattened = list(itertools.chain.from_iterable(time_indexes))
    pairs = [(si, ti) for si, ti in zip(sample_indexes_flattened, time_indexes_flattened)]
    return pairs  # type: ignore


@pydantic.validate_arguments(config={"arbitrary_types_allowed": True, "smart_union": True})
def array3d_to_multiindex_timeseries_dataframe(
    array: np.ndarray,
    *,
    sample_index: data_typing.SampleIndex,
    time_indexes: data_typing.TimeIndexList,
    feature_index: data_typing.FeatureIndex,
    padding_indicator: Any = None,
) -> pd.DataFrame:
    """Given a 3D timeseries `array`, `sample_index`, `time_indexes`, `feature_index`, and a `padding_indicator`, build
    a 2-level multiindex (sample, timestep) `pandas.DataFrame`.

    Padding values of `padding_indicator` can be set inside the array to pad out the length of arrays of different
    samples in case they differ. Padding needs to go at the end of the timesteps (dim 1). Padding must be the same
    across the feature dimension (dim 2) for each sample.

    Raises:
        ValueError: if data or padding format is unexpected.

    Args:
        array (np.ndarray): 3D numpy `array` that represents timeseries like `(sample, timestep, feature)`.
        sample_index (List[<sample element>): List of sample IDs (should be the same length as dim 0 of `array`).
        time_indexes (List[List[<timestep element>]]): List of lists containing timesteps for each sample (outer list
        should be the same length as dim 0 of `array`, inner list should contain as many elements as each sample has
        timesteps).
        feature_index (List[<feature element>]): List of feature names.
        padding_indicator (Any, optional): Padding indicator used in `array` to indicate padding. Defaults to None.

    Returns:
        pd.DataFrame: Resultant dataframe.
    """
    validate_timeseries_array3d(array, padding_indicator)
    unpadded_arrays = unpad_timeseries_array3d(array, padding_indicator)
    data = np.concatenate(unpadded_arrays)
    return pd.DataFrame(
        data,
        index=pd.MultiIndex.from_tuples(make_sample_time_index_tuples(sample_index, time_indexes)),
        columns=feature_index,
    )
