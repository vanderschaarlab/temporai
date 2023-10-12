import copy

import numpy as np
import pandas as pd
import pytest

from tempor.data import dataset, samples

# --- Reusable datasets. ---


# Dummy data: full, with categoricals.
@pytest.fixture(scope="session")
def _dummy_data_with_categorical_features_full() -> dataset.OneOffPredictionDataset:
    from tempor.datasources.prediction.temporal.plugin_dummy_prediction import DummyTemporalPredictionDataSource

    data = DummyTemporalPredictionDataSource(
        static_covariates_missing_prob=0.0,
        temporal_covariates_missing_prob=0.0,
        random_state=777,
    ).load()

    # Add static categorical features.
    static_df = data.static.dataframe()  # pyright: ignore
    np.random.seed(777)
    cat1 = pd.Categorical(np.random.choice(["a", "b", "c"], size=(len(static_df),)))
    cat2 = pd.Categorical(np.random.choice(["D", "E"], size=(len(static_df),)))
    static_df.insert(1, "categorical_feat_1", cat1)  # pyright: ignore
    static_df["categorical_feat_2"] = cat2
    data.static = samples.StaticSamples.from_dataframe(static_df)

    # Add time series categorical features.
    ts_df = data.time_series.dataframe()
    np.random.seed(111)
    cat1 = pd.Categorical(np.random.choice(["p", "q", "r"], size=(len(ts_df),)))
    cat2 = pd.Categorical(np.random.choice(["S", "T"], size=(len(ts_df),)))
    ts_df.insert(1, "categorical_feat_1", cat1)  # pyright: ignore
    ts_df["categorical_feat_2"] = cat2
    data.time_series = samples.TimeSeriesSamples.from_dataframe(ts_df)

    return data  # pyright: ignore


@pytest.fixture(scope="function")
def dummy_data_with_categorical_features_full(
    _dummy_data_with_categorical_features_full: dataset.OneOffPredictionDataset,
) -> dataset.OneOffPredictionDataset:
    # Give each test a copy, just in case.
    return copy.deepcopy(_dummy_data_with_categorical_features_full)


# Dummy data: small, with categoricals.
@pytest.fixture(scope="session")
def _dummy_data_with_categorical_features_small(
    _dummy_data_with_categorical_features_full: dataset.OneOffPredictionDataset,
) -> dataset.OneOffPredictionDataset:
    data = _dummy_data_with_categorical_features_full[:5]
    return data


@pytest.fixture(scope="function")
def dummy_data_with_categorical_features_small(
    _dummy_data_with_categorical_features_small: dataset.OneOffPredictionDataset,
) -> dataset.OneOffPredictionDataset:
    # Give each test a copy, just in case.
    return copy.deepcopy(_dummy_data_with_categorical_features_small)
