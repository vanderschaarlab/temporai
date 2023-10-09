import copy

import pytest

from tempor.data import dataset, datasources

# --- Reusable datasets. ---


# Sine data: full, scaled.
@pytest.fixture(scope="session")
def _sine_data_scaled_full():
    return datasources.SineDataSource(no=100, temporal_dim=5, random_state=42, static_scale=100, ts_scale=100).load()


@pytest.fixture(scope="function")
def sine_data_scaled_full(_sine_data_scaled_full: dataset.OneOffPredictionDataset) -> dataset.OneOffPredictionDataset:
    # Give each test a copy, just in case.
    return copy.deepcopy(_sine_data_scaled_full)


# Sine data: small, scaled.
@pytest.fixture(scope="session")
def _sine_data_scaled_small(_sine_data_scaled_full: dataset.OneOffPredictionDataset) -> dataset.OneOffPredictionDataset:
    data = _sine_data_scaled_full[:6]
    return data


@pytest.fixture(scope="function")
def sine_data_scaled_small(_sine_data_scaled_small: dataset.OneOffPredictionDataset) -> dataset.OneOffPredictionDataset:
    return copy.deepcopy(_sine_data_scaled_small)
