# pylint: disable=redefined-outer-name
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from clairvoyance2.data import Dataset as Clairvoyance2Dataset
from clairvoyance2.data import EventSamples, StaticSamples
from clairvoyance2.datasets import dummy_dataset, simple_pkpd_dataset

from tempor.data import clv2conv, dataset


@pytest.fixture(scope="module")
def pkpd_data():
    return simple_pkpd_dataset(
        n_timesteps=10,
        time_index_treatment_event=5,
        n_control_samples=10,
        n_treated_samples=10,
        seed=100,
    )


class TestClairvoyance2DatasetToTemporDataset:
    def test_fails_no_predictive(self):
        data = dummy_dataset(
            n_samples=10,
            temporal_targets_n_features=0,
            temporal_treatments_n_features=0,
        )

        with pytest.raises(ValueError, match=".*predictive.*"):
            clv2conv.clairvoyance2_dataset_to_tempor_dataset(data)

    def test_fails_event_covariates_present(self):
        data = dummy_dataset(
            n_samples=10,
            temporal_targets_n_features=5,
            temporal_treatments_n_features=0,
        )
        data.event_covariates = Mock()

        with pytest.raises(ValueError, match=".*event_covariates.*"):
            clv2conv.clairvoyance2_dataset_to_tempor_dataset(data)

    def test_fails_both_targets_present(self):
        data = dummy_dataset(
            n_samples=10,
            temporal_targets_n_features=5,
            temporal_treatments_n_features=0,
        )
        data.event_targets = Mock()

        with pytest.raises(ValueError, match=".*temporal_targets.*event_targets.*"):
            clv2conv.clairvoyance2_dataset_to_tempor_dataset(data)

    def test_fails_both_treatments_present(self):
        data = dummy_dataset(
            n_samples=10,
            temporal_targets_n_features=0,
            temporal_treatments_n_features=5,
        )
        data.event_treatments = Mock()

        with pytest.raises(ValueError, match=".*temporal_treatments.*event_treatments.*"):
            clv2conv.clairvoyance2_dataset_to_tempor_dataset(data)

    @pytest.mark.parametrize("n_static", [0, 5])
    def test_time_to_event_analysis_dataset(self, n_static):
        data = dummy_dataset(
            n_samples=10,
            temporal_targets_n_features=0,
            temporal_treatments_n_features=0,
            static_covariates_n_features=n_static,
        )
        np.random.seed(0)
        dummy_event_data = pd.DataFrame(
            data={
                "sample_index": data.temporal_covariates.sample_indices,
                "time_index": list(range(data.n_samples)),
                "feat_0": np.random.randint(low=0, high=2, size=(data.n_samples)),
            }
        )
        dummy_event_data.set_index(keys=["sample_index", "time_index"], drop=True, inplace=True)
        data.event_targets = EventSamples(dummy_event_data)

        data_converted = clv2conv.clairvoyance2_dataset_to_tempor_dataset(data)

        assert isinstance(data_converted, dataset.TimeToEventAnalysisDataset)
        assert len(data_converted) == data.n_samples
        assert data_converted.time_series.sample_index() == data.sample_indices
        assert data_converted.time_series.num_features == data.temporal_covariates.n_features
        assert data_converted.predictive.targets.num_features == data.event_targets.n_features  # pyright: ignore
        if n_static > 0:
            assert data_converted.static.num_features == data.static_covariates.n_features  # pyright: ignore

    @pytest.mark.parametrize("n_static", [0, 5])
    def test_temporal_prediction_dataset(self, n_static):
        data = dummy_dataset(
            n_samples=10,
            temporal_targets_n_features=3,
            temporal_treatments_n_features=0,
            static_covariates_n_features=n_static,
        )

        data_converted = clv2conv.clairvoyance2_dataset_to_tempor_dataset(data)

        assert isinstance(data_converted, dataset.TemporalPredictionDataset)
        assert len(data_converted) == data.n_samples
        assert data_converted.time_series.sample_index() == data.sample_indices
        assert data_converted.time_series.num_features == data.temporal_covariates.n_features
        assert data_converted.predictive.targets.num_features == data.temporal_targets.n_features  # pyright: ignore
        if n_static > 0:
            assert data_converted.static.num_features == data.static_covariates.n_features  # pyright: ignore

    @pytest.mark.parametrize("n_static", [0, 5])
    def test_one_off_treatment_effects_dataset(self, n_static, pkpd_data: Clairvoyance2Dataset):
        data = pkpd_data
        if n_static > 0:
            data.static_covariates = StaticSamples(data=np.ones((data.n_samples, n_static)))
        data_converted = clv2conv.clairvoyance2_dataset_to_tempor_dataset(data)

        assert isinstance(data_converted, dataset.OneOffTreatmentEffectsDataset)
        assert len(data_converted) == data.n_samples
        assert data_converted.time_series.sample_index() == data.sample_indices
        assert data_converted.time_series.num_features == data.temporal_covariates.n_features
        assert data_converted.predictive.targets.num_features == data.temporal_targets.n_features  # pyright: ignore
        assert data_converted.predictive.treatments.num_features == data.event_treatments.n_features  # pyright: ignore
        if n_static > 0:
            assert data_converted.static.num_features == data.static_covariates.n_features  # pyright: ignore

    @pytest.mark.parametrize("n_static", [0, 5])
    def test_temporal_treatment_effects_dataset(self, n_static):
        data = dummy_dataset(
            n_samples=10,
            temporal_targets_n_features=5,
            temporal_treatments_n_features=5,
            static_covariates_n_features=n_static,
        )
        data_converted = clv2conv.clairvoyance2_dataset_to_tempor_dataset(data)

        assert isinstance(data_converted, dataset.TemporalTreatmentEffectsDataset)
        assert len(data_converted) == data.n_samples
        assert data_converted.time_series.sample_index() == data.sample_indices
        assert data_converted.time_series.num_features == data.temporal_covariates.n_features
        assert data_converted.predictive.targets.num_features == data.temporal_targets.n_features  # pyright: ignore
        assert (
            data_converted.predictive.treatments.num_features == data.temporal_treatments.n_features  # pyright: ignore
        )
        if n_static > 0:
            assert data_converted.static.num_features == data.static_covariates.n_features  # pyright: ignore
