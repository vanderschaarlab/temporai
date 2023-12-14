# mypy: ignore-errors

from typing import Dict, List, Sequence, Tuple, TypeVar

import pandas as pd

from ...utils.common import empty_df_like
from ..constants import T_SamplesIndexDtype
from ..dataformat import StaticSamples, TimeSeries, TimeSeriesSamples
from ..dataset import Dataset


def _split_time_series_check_too_short(time_series: TimeSeries, min_len: int, repeat_last_pre_step: bool) -> None:
    actual_min_len = min_len - (1 if repeat_last_pre_step else 0)
    if time_series.n_timesteps < actual_min_len:
        raise ValueError(f"Cannot split a TimeSeries which has a total number of time steps < {actual_min_len}.")


def split(time_series: TimeSeries, at_iloc: int, repeat_last_pre_step: bool = False) -> Tuple[TimeSeries, TimeSeries]:
    _split_time_series_check_too_short(time_series, min_len=2, repeat_last_pre_step=repeat_last_pre_step)
    min_at_iloc, max_at_iloc = 1, time_series.n_timesteps - 1 + (1 if repeat_last_pre_step else 0)
    if not (min_at_iloc <= at_iloc <= max_at_iloc):
        raise ValueError(
            f"Expected `at_iloc` to be in range {[min_at_iloc, max_at_iloc]} "
            f"to split a time series of {time_series.n_timesteps}-many time steps, but `at_iloc` was {at_iloc}"
        )
    pre_df = time_series.df.iloc[:at_iloc]
    post_at_iloc = at_iloc if not repeat_last_pre_step else (at_iloc - 1)
    post_df = time_series.df.iloc[post_at_iloc:]
    assert len(pre_df) + len(post_df) == time_series.n_timesteps + (1 if repeat_last_pre_step else 0)
    return (
        TimeSeries.new_like(like=time_series, data=pre_df),
        TimeSeries.new_like(like=time_series, data=post_df),
    )


TTimeSeries = TypeVar("TTimeSeries", TimeSeries, TimeSeriesSamples, Dataset)


def _split_at_each_step__time_series(
    time_series: TimeSeries,
    min_pre_len: int = 1,
    min_post_len: int = 1,
    repeat_last_pre_step: bool = False,
) -> Tuple[Tuple[TimeSeries, ...], Tuple[TimeSeries, ...], int]:
    _split_time_series_check_too_short(
        time_series, min_len=(min_pre_len + min_post_len), repeat_last_pre_step=repeat_last_pre_step
    )
    min_at_iloc = min_pre_len
    max_at_iloc = time_series.n_timesteps - min_post_len + (1 if repeat_last_pre_step else 0)
    list_pre, list_post = [], []
    for at_iloc in range(min_at_iloc, max_at_iloc + 1):
        ts0, ts1 = split(time_series, at_iloc, repeat_last_pre_step=repeat_last_pre_step)
        list_pre.append(ts0)
        list_post.append(ts1)
    assert len(list_pre) == len(list_post)
    return tuple(list_pre), tuple(list_post), len(list_pre)


TSamplesMap = Dict[T_SamplesIndexDtype, Sequence[int]]


def _split_at_each_step__time_series_samples(
    time_series_samples: TimeSeriesSamples,
    min_pre_len: int = 1,
    min_post_len: int = 1,
    repeat_last_pre_step: bool = False,
) -> Tuple[TimeSeriesSamples, TimeSeriesSamples, TSamplesMap]:
    list_pre: List[TimeSeries] = []
    list_post: List[TimeSeries] = []
    samples_count_map: TSamplesMap = dict()
    counter = 0
    for sample_idx, ts in zip(time_series_samples.sample_indices, time_series_samples):
        pre_, post_, count = _split_at_each_step__time_series(
            ts, min_pre_len=min_pre_len, min_post_len=min_post_len, repeat_last_pre_step=repeat_last_pre_step
        )
        samples_count_map[sample_idx] = list(range(counter, counter + count))
        list_pre.extend(pre_)
        list_post.extend(post_)
        counter += count
    return (
        TimeSeriesSamples.new_like(
            like=time_series_samples,
            data=list_pre,
            sample_indices=None,  # NOTE: Samples are relabelled.
        ),
        TimeSeriesSamples.new_like(
            like=time_series_samples,
            data=list_post,
            sample_indices=None,  # NOTE: Samples are relabelled.
        ),
        samples_count_map,
    )


def _share_out_static_samples(static_samples: StaticSamples, samples_count_map: TSamplesMap) -> pd.DataFrame:
    df = StaticSamples.new_like(like=static_samples, data=static_samples.df).df
    df = empty_df_like(df)

    default_idx = 0
    for old_sample_idx, map_to in samples_count_map.items():
        for _ in map_to:
            with pd.option_context("mode.chained_assignment", None):  # Warning expected.
                df.loc[default_idx, :] = static_samples[old_sample_idx].df.values[0]
            default_idx += 1

    return df


def _split_at_each_step__dataset(
    data: Dataset,
    min_pre_len: int = 1,
    min_post_len: int = 1,
    repeat_last_pre_step: bool = False,
) -> Tuple[Dataset, Dataset, TSamplesMap]:
    # TODO: Doesn't check whether the time index on cov, targ, treat etc. matches.

    data_0_kwargs, data_1_kwargs = dict(), dict()
    samples_count_map = None
    for container_name, container in data.temporal_data_containers.items():
        data_0_container, data_1_container, samples_count_map = _split_at_each_step__time_series_samples(
            container, min_pre_len=min_pre_len, min_post_len=min_post_len, repeat_last_pre_step=repeat_last_pre_step
        )
        data_0_kwargs[container_name] = data_0_container
        data_1_kwargs[container_name] = data_1_container
        if container_name == "temporal_covariates":
            for static_container_name, static_container in data.static_data_containers.items():
                df = _share_out_static_samples(static_container, samples_count_map)
                data_0_kwargs[static_container_name] = df.copy()
                data_1_kwargs[static_container_name] = df.copy()
    assert samples_count_map is not None

    data_0 = Dataset.new_like(
        like=data,
        sample_indices=None,  # NOTE: Samples are relabelled.
        **data_0_kwargs,
    )
    data_1 = Dataset.new_like(
        like=data,
        sample_indices=None,  # NOTE: Samples are relabelled.
        **data_1_kwargs,
    )

    return data_0, data_1, samples_count_map


def split_at_each_step(
    time_series_container: TTimeSeries,
    min_pre_len: int = 1,
    min_post_len: int = 1,
    repeat_last_pre_step: bool = False,
):
    if isinstance(time_series_container, TimeSeries):
        return _split_at_each_step__time_series(
            time_series_container,
            min_pre_len=min_pre_len,
            min_post_len=min_post_len,
            repeat_last_pre_step=repeat_last_pre_step,
        )
    elif isinstance(time_series_container, TimeSeriesSamples):
        return _split_at_each_step__time_series_samples(
            time_series_container,
            min_pre_len=min_pre_len,
            min_post_len=min_post_len,
            repeat_last_pre_step=repeat_last_pre_step,
        )
    elif isinstance(time_series_container, Dataset):
        return _split_at_each_step__dataset(
            time_series_container,
            min_pre_len=min_pre_len,
            min_post_len=min_post_len,
            repeat_last_pre_step=repeat_last_pre_step,
        )
    else:
        raise TypeError(f"Unexpected time series container passed: {type(time_series_container)}")
