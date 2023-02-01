# pylint: disable=redefined-outer-name, unused-argument

import pytest

import tempor.data.samples as s


@pytest.fixture
def df_static(df_static_cat_num_hasnan):
    return df_static_cat_num_hasnan


@pytest.fixture
def df_time_series(df_time_series_num_nonan):
    return df_time_series_num_nonan


@pytest.fixture
def df_event(df_event_num_nonan):
    return df_event_num_nonan


def test_instantiation_success(df_static, df_time_series, df_event):
    _ = s.StaticSamples(data=df_static)
    _ = s.TimeSeriesSamples(data=df_time_series)
    _ = s.EventSamples(data=df_event)
