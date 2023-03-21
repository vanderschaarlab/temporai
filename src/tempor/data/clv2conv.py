"""Utilities for converting to and from ``clairvoyance2`` datasets."""
from typing import Type

import pandas as pd
from clairvoyance2.data import Dataset as Clairvoyance2Dataset

from tempor.data import dataset

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


def clairvoyance2_dataset_to_tempor_dataset(data: Clairvoyance2Dataset) -> dataset.Dataset:
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
    else:
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
    if data.temporal_treatments is not None:
        treatments_df = _from_clv2_time_series(treatments_df)  # pyright: ignore
    elif data.event_treatments is not None:
        treatments_df = _from_clv2_event(treatments_df)  # pyright: ignore

    # Determine tempor.Dataset subclass.
    if data.temporal_targets is not None and data.temporal_treatments is None and data.event_treatments is None:
        TemporDatasetCls: Type[dataset.Dataset] = dataset.TemporalPredictionDataset
    elif data.event_targets is not None and data.temporal_treatments is None and data.event_treatments is None:
        TemporDatasetCls = dataset.TimeToEventAnalysisDataset
    elif data.temporal_targets is not None and data.event_treatments is not None:
        TemporDatasetCls = dataset.OneOffTreatmentEffectsDataset
    elif data.temporal_targets is not None and data.temporal_treatments is not None:
        TemporDatasetCls = dataset.TemporalTreatmentEffectsDataset
    else:
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
