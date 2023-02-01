# pylint: disable=redefined-outer-name, unused-argument
# TODO: Test more root validation cases.

import pandas as pd
import pytest

import tempor.data.container._requirements as dr
import tempor.data.types as types
import tempor.exc
from tempor.data.container._validator.impl import df_validator


@pytest.fixture
def df_static(df_static_cat_num_hasnan):
    return df_static_cat_num_hasnan


@pytest.fixture
def df_time_series(df_time_series_num_nonan):
    return df_time_series_num_nonan


@pytest.fixture
def df_event(df_event_num_nonan):
    return df_event_num_nonan


def test_static_validator_root_validation(df_static):
    validator = df_validator.StaticDataValidator()
    validator.validate(df_static, requirements=[], container_flavor=types.ContainerFlavor.DF_SAMPLE_X_FEATURE)


def test_common_requirements_passes(df_static):
    validator = df_validator.StaticDataValidator()
    validator.validate(
        df_static,
        requirements=[dr.ValueDTypes([float, "category"])],
        container_flavor=types.ContainerFlavor.DF_SAMPLE_X_FEATURE,
    )
    validator.validate(
        df_static,
        requirements=[dr.AllowMissing(definition=True)],
        container_flavor=types.ContainerFlavor.DF_SAMPLE_X_FEATURE,
    )


def test_common_requirements_fails(df_static):
    validator = df_validator.StaticDataValidator()

    with pytest.raises(tempor.exc.DataValidationFailedException):
        validator.validate(
            df_static,
            requirements=[dr.ValueDTypes(definition=[bool])],
            container_flavor=types.ContainerFlavor.DF_SAMPLE_X_FEATURE,
        )

    with pytest.raises(tempor.exc.DataValidationFailedException):
        validator.validate(
            df_static,
            requirements=[dr.AllowMissing(definition=False)],
            container_flavor=types.ContainerFlavor.DF_SAMPLE_X_FEATURE,
        )


def test_time_series_validator_root_validation(df_time_series):
    validator = df_validator.TimeSeriesDataValidator()
    validator.validate(
        df_time_series, requirements=[], container_flavor=types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE
    )


def test_event_validator_root_validation(df_event):
    validator = df_validator.EventDataValidator()
    validator.validate(df_event, requirements=[], container_flavor=types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE)


def test_event_validator_root_validation_fails_not_one_to_one_indices():
    df = pd.DataFrame(
        {
            "sample_idx": ["a", "a", "b", "c"],
            "time_idx": [1, 3, 2, 2],
            "f1": [True, True, False, True],
            "f2": [0, 0, 0, 1],
        }
    )
    df = df.set_index(keys=["sample_idx", "time_idx"])

    validator = df_validator.EventDataValidator()

    with pytest.raises(tempor.exc.DataValidationFailedException) as excinfo:
        validator.validate(df, requirements=[], container_flavor=types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE)
    assert "one-to-one" in str(excinfo.getrepr())
