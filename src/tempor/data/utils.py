from typing import Any, Optional

import numpy as np
import pandas as pd


def value_in_df(df: pd.DataFrame, /, *, value: Any) -> bool:
    return (np.isnan(value) and df.isna().any().any()) or (df == value).any().any()


def multiindex_timeseries_df_to_array(
    df: pd.DataFrame, padding_indicator: Any, max_timesteps: Optional[int] = None
) -> np.ndarray:
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
