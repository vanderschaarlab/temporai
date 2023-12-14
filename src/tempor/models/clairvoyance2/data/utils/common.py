from ...data import TimeSeriesSamples


def cast_time_series_samples_feature_names_to_str(time_series_samples: TimeSeriesSamples) -> None:
    for ts in time_series_samples:
        ts.df.rename(columns={c: str(c) for c in ts.df.columns}, inplace=True)
