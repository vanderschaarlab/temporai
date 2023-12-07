"""Utilities for converting to and from ``clairvoyance2`` datasets."""

from typing import List, Type

import pandas as pd
from clairvoyance2.data import Dataset as Clairvoyance2Dataset

from tempor.data import dataset, samples

from . import utils


def _from_clv2_static(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.set_df_column_names_inplace(df, names=[str(c) for c in df.columns])
    df.index.name = "sample_idx"
    return df


def _from_clv2_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df = utils.set_df_column_names_inplace(df, names=[str(c) for c in df.columns])
    df.index.names = ["sample_idx", "time_idx"]
    return df


def _from_clv2_event(df: pd.DataFrame) -> pd.DataFrame:
    column_names = [str(c) for c in df.columns]
    sample_index = df.index.get_level_values(0)
    sample_index.name = "sample_idx"
    event_times = df.index.get_level_values(1)
    data_ = {k: list(zip(event_times, df[k].astype(bool))) for k in column_names}
    df_out = pd.DataFrame(data=data_, index=sample_index, columns=column_names)
    return df_out


def clairvoyance2_dataset_to_tempor_dataset(data: Clairvoyance2Dataset) -> dataset.BaseDataset:
    """A utility function to convert a ``clairvoyance2`` dataset to a TemporAI dataset.

    Args:
        data (Clairvoyance2Dataset): The ``clairvoyance2`` dataset to convert.

    Returns:
        dataset.BaseDataset: The converted dataset.
    """
    if (
        data.temporal_targets is None
        and data.temporal_treatments is None
        and data.event_targets is None
        and data.event_treatments is None
    ):
        raise ValueError(
            "`clairvoyance2` dataset did not have any predictive data (targets or treatments), "
            "this case is not supported"
        )
    if data.temporal_targets is not None and data.event_targets is not None:
        raise ValueError(
            "`clairvoyance2` dataset had both `temporal_targets` and `event_targets`, this case is not supported"
        )
    if data.temporal_treatments is not None and data.event_treatments is not None:
        raise ValueError(
            "`clairvoyance2` dataset had both `temporal_treatments` and `event_treatments`, this case is not supported"
        )
    if data.event_covariates is not None:
        raise ValueError("`clairvoyance2` dataset had `event_covariates`, this case is not supported.")

    # Covariates.
    static_df = data.static_covariates.df if data.static_covariates is not None else None
    time_series_df = data.temporal_covariates.to_multi_index_dataframe()

    # Targets.
    if data.temporal_targets is not None:
        targets_df = data.temporal_targets.to_multi_index_dataframe()
    elif data.event_targets is not None:
        targets_df = data.event_targets.df
    else:  # pragma: no cover
        # Case caught by exceptions above.
        targets_df = None

    # Treatments.
    if data.temporal_treatments is not None:
        treatments_df = data.temporal_treatments.to_multi_index_dataframe()
    elif data.event_treatments is not None:
        treatments_df = data.event_treatments.df
    else:
        treatments_df = None

    # Conversions.
    if static_df is not None:
        static_df = _from_clv2_static(static_df)
    time_series_df = _from_clv2_time_series(time_series_df)
    if data.temporal_targets is not None:
        targets_df = _from_clv2_time_series(targets_df)  # pyright: ignore
    elif data.event_targets is not None:
        targets_df = _from_clv2_event(targets_df)  # pyright: ignore
    else:  # pragma: no cover
        # Case caught by exceptions above.
        targets_df = None
    if data.temporal_treatments is not None:
        treatments_df = _from_clv2_time_series(treatments_df)  # pyright: ignore
    elif data.event_treatments is not None:
        treatments_df = _from_clv2_event(treatments_df)  # pyright: ignore

    # Determine tempor.Dataset subclass.
    if data.temporal_targets is not None and data.temporal_treatments is None and data.event_treatments is None:
        TemporDatasetCls: Type[dataset.BaseDataset] = dataset.TemporalPredictionDataset
    elif data.event_targets is not None and data.temporal_treatments is None and data.event_treatments is None:
        TemporDatasetCls = dataset.TimeToEventAnalysisDataset
    elif data.temporal_targets is not None and data.event_treatments is not None:
        TemporDatasetCls = dataset.OneOffTreatmentEffectsDataset
    elif data.temporal_targets is not None and data.temporal_treatments is not None:
        TemporDatasetCls = dataset.TemporalTreatmentEffectsDataset
    else:  # pragma: no cover
        raise ValueError(
            "Cannot convert a clairvoyance2 dataset to tempor dataset in this case, as it is "
            f"not supported, see clairvoyance2 dataset components:\n{data}"
        )

    return TemporDatasetCls(
        time_series=time_series_df,
        static=static_df,
        targets=targets_df,  # pyright: ignore
        treatments=treatments_df,  # pyright: ignore
    )


def _to_clv2_static(s: samples.StaticSamplesBase) -> pd.DataFrame:
    int_sample_index = list(range(s.num_samples))
    return s.dataframe().set_index(keys=pd.Index(int_sample_index), drop=True)


def _to_clv2_time_series(s: samples.TimeSeriesSamplesBase) -> List[pd.DataFrame]:
    return [df.droplevel(0) for df in s.list_of_dataframes()]


def _to_clv2_event(s: samples.EventSamplesBase) -> pd.DataFrame:
    int_sample_index = list(range(s.num_samples))

    df_event_times, df_event_values = s.split_as_two_dataframes()

    all_event_times_match = df_event_times.eq(df_event_times.iloc[:, 0], axis=0).all(1).all()
    # ^ Check all time columns equal else exception.
    if not all_event_times_match:
        raise ValueError(
            "Event times must be the same for all features of each sample in order to "
            "be convertible to a clairvoyance2 dataset"
        )
    times = df_event_times.iloc[:, 0].to_list()

    df = df_event_values.astype(int).set_index(keys=[pd.Index(int_sample_index), times])  # pyright: ignore

    return df


def tempor_dataset_to_clairvoyance2_dataset(data: dataset.BaseDataset) -> Clairvoyance2Dataset:
    """A utility function to convert a TemporAI dataset to a ``clairvoyance2`` dataset.

    Args:
        data (dataset.BaseDataset): The TemporAI dataset to convert.

    Returns:
        Clairvoyance2Dataset: The converted dataset.
    """
    if isinstance(data, dataset.OneOffPredictionDataset):
        raise ValueError(
            "Cannot convert a `OneOffPredictionDataset` to a clairvoyance2 dataset, as this setting is not supported"
        )

    def has_temporal_targets(d: dataset.BaseDataset) -> bool:
        if d.predictive is not None:
            return isinstance(d.predictive.targets, samples.TimeSeriesSamplesBase)
        else:
            return False

    def has_temporal_treatments(d: dataset.BaseDataset) -> bool:
        if d.predictive is not None:
            return isinstance(d.predictive.treatments, samples.TimeSeriesSamplesBase)
        else:
            return False

    def has_event_targets(d: dataset.BaseDataset) -> bool:
        if d.predictive is not None:
            return isinstance(d.predictive.targets, samples.EventSamplesBase)
        else:
            return False

    def has_event_treatments(d: dataset.BaseDataset) -> bool:
        if d.predictive is not None:
            return isinstance(d.predictive.treatments, samples.EventSamplesBase)
        else:
            return False

    return Clairvoyance2Dataset(
        temporal_covariates=_to_clv2_time_series(data.time_series),
        static_covariates=_to_clv2_static(data.static) if data.static is not None else None,
        event_covariates=None,
        temporal_targets=(
            _to_clv2_time_series(data.predictive.targets) if has_temporal_targets(data) else None  # type: ignore
        ),
        temporal_treatments=(
            _to_clv2_time_series(data.predictive.treatments) if has_temporal_treatments(data) else None  # type: ignore
        ),
        event_targets=(_to_clv2_event(data.predictive.targets) if has_event_targets(data) else None),  # type: ignore
        event_treatments=(
            _to_clv2_event(data.predictive.treatments) if has_event_treatments(data) else None  # type: ignore
        ),
    )
