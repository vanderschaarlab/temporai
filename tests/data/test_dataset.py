# pylint: disable=redefined-outer-name, unused-argument, protected-access

import dataclasses
from typing import TYPE_CHECKING, Tuple, Type
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import sklearn.model_selection

from tempor.data import dataset, predictive, samples

# --- Heavily mocked unit tests. ---


@dataclasses.dataclass(frozen=True)
class MockedDepends:
    StaticSamples: Mock
    TimeSeriesSamples: Mock
    EventSamples: Mock

    # predictive:
    OneOffPredictionTaskData: Mock
    TemporalPredictionTaskData: Mock
    TimeToEventAnalysisTaskData: Mock
    OneOffTreatmentEffectsTaskData: Mock
    TemporalTreatmentEffectsTaskData: Mock


@pytest.fixture
def mocked_depends(monkeypatch) -> MockedDepends:
    MockStaticSamples = Mock()
    MockTimeSeriesSamples = Mock()
    MockEventSamples = Mock()
    monkeypatch.setattr(samples, "StaticSamples", MockStaticSamples)
    monkeypatch.setattr(samples, "TimeSeriesSamples", MockTimeSeriesSamples)
    monkeypatch.setattr(samples, "EventSamples", MockEventSamples)

    MockOneOffPredictionTaskData = Mock()
    MockTemporalPredictionTaskData = Mock()
    MockTimeToEventAnalysisTaskData = Mock()
    MockOneOffTreatmentEffectsTaskData = Mock()
    MockTemporalTreatmentEffectsTaskData = Mock()
    monkeypatch.setattr(predictive, "OneOffPredictionTaskData", MockOneOffPredictionTaskData)
    monkeypatch.setattr(predictive, "TemporalPredictionTaskData", MockTemporalPredictionTaskData)
    monkeypatch.setattr(predictive, "TimeToEventAnalysisTaskData", MockTimeToEventAnalysisTaskData)
    monkeypatch.setattr(predictive, "OneOffTreatmentEffectsTaskData", MockOneOffTreatmentEffectsTaskData)
    monkeypatch.setattr(predictive, "TemporalTreatmentEffectsTaskData", MockTemporalTreatmentEffectsTaskData)

    return MockedDepends(
        StaticSamples=MockStaticSamples,
        TimeSeriesSamples=MockTimeSeriesSamples,
        EventSamples=MockEventSamples,
        # predictive:
        OneOffPredictionTaskData=MockOneOffPredictionTaskData,
        TemporalPredictionTaskData=MockTemporalPredictionTaskData,
        TimeToEventAnalysisTaskData=MockTimeToEventAnalysisTaskData,
        OneOffTreatmentEffectsTaskData=MockOneOffTreatmentEffectsTaskData,
        TemporalTreatmentEffectsTaskData=MockTemporalTreatmentEffectsTaskData,
    )


def mock_indexes(monkeypatch, cls_name, sample_index, time_indexes):
    instance = Mock()
    instance.sample_index = Mock(return_value=sample_index)
    instance.time_indexes = Mock(return_value=time_indexes)
    monkeypatch.setattr(samples, cls_name, Mock(return_value=instance))


def mock_predictive_data_indexes(
    monkeypatch,
    cls_name,
    targets_sample_index,
    treatments_sample_index,
    targets_time_indexes=None,
    treatments_time_indexes=None,
):
    targets_instance = Mock()
    targets_instance.sample_index = Mock(return_value=targets_sample_index)
    if targets_time_indexes is not None:
        targets_instance.time_indexes = Mock(return_value=targets_time_indexes)

    treatments_instance = Mock()
    treatments_instance.sample_index = Mock(return_value=treatments_sample_index)
    if treatments_time_indexes is not None:
        treatments_instance.time_indexes = Mock(return_value=treatments_time_indexes)

    MockPredictionTaskData = Mock(return_value=Mock(targets=targets_instance, treatments=treatments_instance))
    monkeypatch.setattr(predictive, cls_name, MockPredictionTaskData)


# --- Dataset base class. ---


class MockedBaseDataset(dataset.BaseDataset):
    mock_validate_call: Mock
    mock_init_predictive: Mock

    def _init_predictive(self, targets, treatments=None, **kwargs) -> None:
        self.predictive = self.mock_init_predictive(targets=targets, treatments=treatments, **kwargs)
        self.predictive.predictive_task = "dummy_task"

    def validate(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mock_validate_call()


@pytest.fixture
def mock_dataset_cls():
    MockedBaseDataset.mock_validate_call = Mock()
    MockedBaseDataset.mock_init_predictive = Mock()
    return MockedBaseDataset


class TestBaseDataset:
    def test_init(self, mock_dataset_cls: Type[MockedBaseDataset], mocked_depends: MockedDepends):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = Mock()
        data_treatments = Mock()

        dummy_dataset = mock_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            treatments=data_treatments,
            **{"dummy": "kwarg"},
        )

        mocked_depends.TimeSeriesSamples.assert_called_once_with(data_time_series)
        mocked_depends.StaticSamples.assert_called_once_with(data_static)
        dummy_dataset.mock_init_predictive.assert_called_once_with(
            targets=data_targets, treatments=data_treatments, dummy="kwarg"
        )
        dummy_dataset.mock_validate_call.assert_called_once()

    def test_init_no_targets_provided(self, mock_dataset_cls: Type[MockedBaseDataset], mocked_depends: MockedDepends):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = None

        dummy_dataset = mock_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            **{"dummy": "kwarg"},
        )

        mocked_depends.TimeSeriesSamples.assert_called_once_with(data_time_series)
        mocked_depends.StaticSamples.assert_called_once_with(data_static)
        dummy_dataset.mock_init_predictive.assert_called_once_with(targets=None, treatments=None, dummy="kwarg")
        dummy_dataset.mock_validate_call.assert_called_once()

    @pytest.mark.parametrize(
        "static, expected",
        [
            (Mock(), True),
            (None, False),
        ],
    )
    def test_has_static(
        self, static, expected: bool, mock_dataset_cls: Type[MockedBaseDataset], mocked_depends: MockedDepends
    ):
        data_time_series = Mock()
        data_targets = None

        dummy_dataset = mock_dataset_cls(
            time_series=data_time_series,
            static=static,
            targets=data_targets,
        )

        assert dummy_dataset.has_static is expected

    @pytest.mark.parametrize(
        "targets, expected_has_predictive_data, expected_predictive_task",
        [
            (Mock(), True, "dummy_task"),
            (None, True, "dummy_task"),
        ],
    )
    def test_has_predictive_data_and_predictive_task(
        self,
        targets,
        expected_has_predictive_data: bool,
        expected_predictive_task,
        mock_dataset_cls: Type[MockedBaseDataset],
        mocked_depends: MockedDepends,
    ):
        data_time_series = Mock()
        data_static = Mock()

        dummy_dataset = mock_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=targets,
        )

        assert dummy_dataset.has_predictive_data is expected_has_predictive_data
        assert dummy_dataset.predictive_task is expected_predictive_task

    @pytest.mark.parametrize("static", [Mock(), None])
    def test_validate_passes(self, static, mock_dataset_cls: Type[MockedBaseDataset], monkeypatch):
        data_time_series = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="StaticSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )

        mock_dataset_cls.validate = dataset.BaseDataset.validate  # type: ignore  # Enable validation code.

        dummy_dataset = mock_dataset_cls(
            time_series=data_time_series,
            static=static,
            targets=None,
        )

        dummy_dataset.mock_validate_call.assert_called_once()

    def test_validate_fails_sample_index_mismatch(self, mock_dataset_cls: Type[MockedBaseDataset], monkeypatch):
        data_time_series = Mock()
        data_static = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="StaticSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2", "s3"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )

        mock_dataset_cls.validate = dataset.BaseDataset.validate  # type: ignore  # Enable validation code.

        with pytest.raises(ValueError, match=".*sample_index.*static.*time series.*"):
            mock_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=None,
            )


# --- OneOffPredictionDataset. ---


class MockedOneOffPredictionDataset(dataset.OneOffPredictionDataset):
    mock_validate_call: Mock

    def validate(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mock_validate_call()


@pytest.fixture
def mock_one_off_prediction_dataset_cls():
    MockedOneOffPredictionDataset.mock_validate_call = Mock()
    return MockedOneOffPredictionDataset


class TestOneOffPredictionDataset:
    def test_init(
        self, mock_one_off_prediction_dataset_cls: Type[MockedOneOffPredictionDataset], mocked_depends: MockedDepends
    ):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = Mock()

        dummy_dataset = mock_one_off_prediction_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            **{"dummy": "kwarg"},
        )

        mocked_depends.TimeSeriesSamples.assert_called_once_with(data_time_series)
        mocked_depends.StaticSamples.assert_called_once_with(data_static)
        mocked_depends.OneOffPredictionTaskData.assert_called_once_with(
            parent_dataset=dummy_dataset, targets=data_targets, dummy="kwarg"
        )
        dummy_dataset.mock_validate_call.assert_called_once()

        assert dummy_dataset.fit_ready
        assert dummy_dataset.predict_ready

    def test_validate_passes(
        self, mock_one_off_prediction_dataset_cls: Type[MockedOneOffPredictionDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="OneOffPredictionTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=None,
        )

        mock_one_off_prediction_dataset_cls._validate = dataset.OneOffPredictionDataset._validate  # type: ignore
        # ^ Enable validation code.

        mock_one_off_prediction_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
        )

    def test_validate_fails_sample_index_mismatch(
        self, mock_one_off_prediction_dataset_cls: Type[MockedOneOffPredictionDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="OneOffPredictionTaskData",
            targets_sample_index=["s1", "s2", "s3"],
            treatments_sample_index=None,
        )

        mock_one_off_prediction_dataset_cls._validate = dataset.OneOffPredictionDataset._validate  # type: ignore
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=".*sample_index.*targets.*time series.*"):
            mock_one_off_prediction_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
            )


# --- TemporalPredictionDataset. ---


class MockedTemporalPredictionDataset(dataset.TemporalPredictionDataset):
    mock_validate_call: Mock

    def validate(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mock_validate_call()


@pytest.fixture
def mock_temporal_prediction_dataset_cls():
    MockedTemporalPredictionDataset.mock_validate_call = Mock()
    return MockedTemporalPredictionDataset


class TestTemporalPredictionDataset:
    def test_init(
        self, mock_temporal_prediction_dataset_cls: Type[MockedTemporalPredictionDataset], mocked_depends: MockedDepends
    ):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = Mock()

        dummy_dataset = mock_temporal_prediction_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            **{"dummy": "kwarg"},
        )

        mocked_depends.TimeSeriesSamples.assert_called_once_with(data_time_series)
        mocked_depends.StaticSamples.assert_called_once_with(data_static)
        mocked_depends.TemporalPredictionTaskData.assert_called_once_with(
            parent_dataset=dummy_dataset, targets=data_targets, dummy="kwarg"
        )
        dummy_dataset.mock_validate_call.assert_called_once()

        assert dummy_dataset.fit_ready
        assert dummy_dataset.predict_ready

    def test_validate_passes(
        self, mock_temporal_prediction_dataset_cls: Type[MockedTemporalPredictionDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="TemporalPredictionTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=None,
            targets_time_indexes=[[1, 2, 3], [1, 2]],
            treatments_time_indexes=None,
        )

        mock_temporal_prediction_dataset_cls._validate = dataset.TemporalPredictionDataset._validate  # type: ignore
        # ^ Enable validation code.

        mock_temporal_prediction_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
        )

    def test_validate_fails_sample_index_mismatch(
        self, mock_temporal_prediction_dataset_cls: Type[MockedTemporalPredictionDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="TemporalPredictionTaskData",
            targets_sample_index=["s1", "s2", "s3"],
            treatments_sample_index=None,
            targets_time_indexes=[[1, 2, 3], [1, 2]],
            treatments_time_indexes=None,
        )

        mock_temporal_prediction_dataset_cls._validate = dataset.TemporalPredictionDataset._validate  # type: ignore
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=".*sample_index.*targets.*time series.*"):
            mock_temporal_prediction_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
            )

    def test_validate_fails_time_indexes_mismatch(
        self, mock_temporal_prediction_dataset_cls: Type[MockedTemporalPredictionDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="TemporalPredictionTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=None,
            targets_time_indexes=[[0, 5, 8], [123]],
            treatments_time_indexes=None,
        )

        mock_temporal_prediction_dataset_cls._validate = dataset.TemporalPredictionDataset._validate  # type: ignore
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=".*time_indexes.*targets.*time series.*"):
            mock_temporal_prediction_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
            )


# --- TimeToEventAnalysisDataset. ---


class MockedTimeToEventAnalysisDataset(dataset.TimeToEventAnalysisDataset):
    mock_validate_call: Mock

    def validate(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mock_validate_call()


@pytest.fixture
def mock_time_to_event_analysis_dataset_cls():
    MockedTimeToEventAnalysisDataset.mock_validate_call = Mock()
    return MockedTimeToEventAnalysisDataset


class TestTimeToEventAnalysisDataset:
    def test_init(
        self,
        mock_time_to_event_analysis_dataset_cls: Type[MockedTimeToEventAnalysisDataset],
        mocked_depends: MockedDepends,
    ):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = Mock()

        dummy_dataset = mock_time_to_event_analysis_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            **{"dummy": "kwarg"},
        )

        mocked_depends.TimeSeriesSamples.assert_called_once_with(data_time_series)
        mocked_depends.StaticSamples.assert_called_once_with(data_static)
        mocked_depends.TimeToEventAnalysisTaskData.assert_called_once_with(
            parent_dataset=dummy_dataset, targets=data_targets, dummy="kwarg"
        )
        dummy_dataset.mock_validate_call.assert_called_once()

        assert dummy_dataset.fit_ready
        assert dummy_dataset.predict_ready

    def test_validate_passes(
        self, mock_time_to_event_analysis_dataset_cls: Type[MockedTimeToEventAnalysisDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="TimeToEventAnalysisTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=None,
        )

        mock_time_to_event_analysis_dataset_cls._validate = dataset.TimeToEventAnalysisDataset._validate  # type: ignore
        # ^ Enable validation code.

        mock_time_to_event_analysis_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
        )

    def test_validate_fails_sample_index_mismatch(
        self, mock_time_to_event_analysis_dataset_cls: Type[MockedTimeToEventAnalysisDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="TimeToEventAnalysisTaskData",
            targets_sample_index=["s1", "s2", "s3"],
            treatments_sample_index=None,
        )

        mock_time_to_event_analysis_dataset_cls._validate = dataset.TimeToEventAnalysisDataset._validate  # type: ignore
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=".*sample_index.*targets.*time series.*"):
            mock_time_to_event_analysis_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
            )


# --- OneOffTreatmentEffectsDataset. ---


class MockedOneOffTreatmentEffectsDataset(dataset.OneOffTreatmentEffectsDataset):
    mock_validate_call: Mock

    def validate(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mock_validate_call()


@pytest.fixture
def mock_one_off_treatment_effects_dataset_cls():
    MockedOneOffTreatmentEffectsDataset.mock_validate_call = Mock()
    return MockedOneOffTreatmentEffectsDataset


class TestOneOffTreatmentEffectsDataset:
    def test_init(
        self,
        mock_one_off_treatment_effects_dataset_cls: Type[MockedOneOffTreatmentEffectsDataset],
        mocked_depends: MockedDepends,
    ):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = Mock()
        data_treatments = Mock()

        dummy_dataset = mock_one_off_treatment_effects_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            treatments=data_treatments,
            **{"dummy": "kwarg"},
        )

        mocked_depends.TimeSeriesSamples.assert_called_once_with(data_time_series)
        mocked_depends.StaticSamples.assert_called_once_with(data_static)
        mocked_depends.OneOffTreatmentEffectsTaskData.assert_called_once_with(
            parent_dataset=dummy_dataset, targets=data_targets, treatments=data_treatments, dummy="kwarg"
        )
        dummy_dataset.mock_validate_call.assert_called_once()

        assert dummy_dataset.fit_ready
        assert dummy_dataset.predict_ready

    def test_init_fails_treatment_not_set(
        self,
        mock_one_off_treatment_effects_dataset_cls: Type[MockedOneOffTreatmentEffectsDataset],
        mocked_depends: MockedDepends,
    ):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = Mock()

        with pytest.raises(ValueError, match=".*requires.*treatment.*"):
            mock_one_off_treatment_effects_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
                treatments=None,  # pyright: ignore
            )

    def test_validate_passes(
        self, mock_one_off_treatment_effects_dataset_cls: Type[MockedOneOffTreatmentEffectsDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        data_treatments = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="OneOffTreatmentEffectsTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=["s1", "s2"],
            targets_time_indexes=[[1, 2, 3], [1, 2]],
            treatments_time_indexes=None,
        )

        mock_one_off_treatment_effects_dataset_cls._validate = (  # type: ignore
            dataset.OneOffTreatmentEffectsDataset._validate
        )
        # ^ Enable validation code.

        mock_one_off_treatment_effects_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            treatments=data_treatments,
        )

    @pytest.mark.parametrize(
        "targets_sample_index, treatments_sample_index, match_exc",
        [
            (["s1", "s2", "s3"], ["s1", "s2"], ".*sample_index.*targets.*time series.*"),
            (["s1", "s2"], ["s1", "s2", "s3"], ".*sample_index.*treatments.*time series.*"),
        ],
    )
    def test_validate_fails_sample_index_mismatch(
        self,
        mock_one_off_treatment_effects_dataset_cls: Type[MockedOneOffTreatmentEffectsDataset],
        monkeypatch,
        targets_sample_index,
        treatments_sample_index,
        match_exc,
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        data_treatments = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="OneOffTreatmentEffectsTaskData",
            targets_sample_index=targets_sample_index,
            treatments_sample_index=treatments_sample_index,
            targets_time_indexes=[[1, 2, 3], [1, 2]],
            treatments_time_indexes=None,
        )

        mock_one_off_treatment_effects_dataset_cls._validate = (  # type: ignore
            dataset.OneOffTreatmentEffectsDataset._validate
        )
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=match_exc):
            mock_one_off_treatment_effects_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
                treatments=data_treatments,
            )

    def test_validate_fails_time_series_and_targets_time_indexes_mismatch(
        self, mock_one_off_treatment_effects_dataset_cls: Type[MockedOneOffTreatmentEffectsDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        data_treatments = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="OneOffTreatmentEffectsTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=["s1", "s2"],
            targets_time_indexes=[[0, 5, 8], [123]],
            treatments_time_indexes=None,
        )

        mock_one_off_treatment_effects_dataset_cls._validate = (  # type: ignore
            dataset.OneOffTreatmentEffectsDataset._validate
        )
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=".*time_indexes.*targets.*time series.*"):
            mock_one_off_treatment_effects_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
                treatments=data_treatments,
            )


# --- TemporalTreatmentEffectsDataset. ---


class MockedTemporalTreatmentEffectsDataset(dataset.TemporalTreatmentEffectsDataset):
    mock_validate_call: Mock

    def validate(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.mock_validate_call()


@pytest.fixture
def mock_temporal_treatment_effects_dataset_cls():
    MockedTemporalTreatmentEffectsDataset.mock_validate_call = Mock()
    return MockedTemporalTreatmentEffectsDataset


class TestTemporalTreatmentEffectsDataset:
    def test_init(
        self,
        mock_temporal_treatment_effects_dataset_cls: Type[MockedTemporalTreatmentEffectsDataset],
        mocked_depends: MockedDepends,
    ):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = Mock()
        data_treatments = Mock()

        dummy_dataset = mock_temporal_treatment_effects_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            treatments=data_treatments,
            **{"dummy": "kwarg"},
        )

        mocked_depends.TimeSeriesSamples.assert_called_once_with(data_time_series)
        mocked_depends.StaticSamples.assert_called_once_with(data_static)
        mocked_depends.TemporalTreatmentEffectsTaskData.assert_called_once_with(
            parent_dataset=dummy_dataset, targets=data_targets, treatments=data_treatments, dummy="kwarg"
        )
        dummy_dataset.mock_validate_call.assert_called_once()

        assert dummy_dataset.fit_ready
        assert dummy_dataset.predict_ready

    def test_init_fails_treatment_not_set(
        self,
        mock_temporal_treatment_effects_dataset_cls: Type[MockedTemporalTreatmentEffectsDataset],
        mocked_depends: MockedDepends,
    ):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = Mock()

        with pytest.raises(ValueError, match=".*requires.*treatment.*"):
            mock_temporal_treatment_effects_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
                treatments=None,  # pyright: ignore
            )

    def test_validate_passes(
        self, mock_temporal_treatment_effects_dataset_cls: Type[MockedTemporalTreatmentEffectsDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        data_treatments = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="TemporalTreatmentEffectsTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=["s1", "s2"],
            targets_time_indexes=[[1, 2, 3], [1, 2]],
            treatments_time_indexes=[[1, 2, 3], [1, 2]],
        )

        mock_temporal_treatment_effects_dataset_cls._validate = (  # type: ignore
            dataset.TemporalTreatmentEffectsDataset._validate
        )
        # ^ Enable validation code.

        mock_temporal_treatment_effects_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
            treatments=data_treatments,
        )

    @pytest.mark.parametrize(
        "targets_sample_index, treatments_sample_index, match_exc",
        [
            (["s1", "s2", "s3"], ["s1", "s2"], ".*sample_index.*targets.*time series.*"),
            (["s1", "s2"], ["s1", "s2", "s3"], ".*sample_index.*treatments.*time series.*"),
        ],
    )
    def test_validate_fails_sample_index_mismatch(
        self,
        mock_temporal_treatment_effects_dataset_cls: Type[MockedTemporalTreatmentEffectsDataset],
        monkeypatch,
        targets_sample_index,
        treatments_sample_index,
        match_exc,
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        data_treatments = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="TemporalTreatmentEffectsTaskData",
            targets_sample_index=targets_sample_index,
            treatments_sample_index=treatments_sample_index,
            targets_time_indexes=[[1, 2, 3], [1, 2]],
            treatments_time_indexes=[[1, 2, 3], [1, 2]],
        )

        mock_temporal_treatment_effects_dataset_cls._validate = (  # type: ignore
            dataset.TemporalTreatmentEffectsDataset._validate
        )
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=match_exc):
            mock_temporal_treatment_effects_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
                treatments=data_treatments,
            )

    @pytest.mark.parametrize(
        "targets_time_indexes, treatments_time_indexes, match_exc",
        [
            ([[0, 5, 8], [123]], [[1, 2, 3], [1, 2]], ".*time_indexes.*targets.*time series.*"),
            ([[1, 2, 3], [1, 2]], [[0, 5, 8], [123]], ".*time_indexes.*treatments.*time series.*"),
        ],
    )
    def test_validate_fails_time_indexes_mismatch(
        self,
        mock_temporal_treatment_effects_dataset_cls: Type[MockedTemporalTreatmentEffectsDataset],
        monkeypatch,
        targets_time_indexes,
        treatments_time_indexes,
        match_exc,
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        data_treatments = Mock()
        mock_indexes(
            monkeypatch,
            cls_name="TimeSeriesSamples",
            sample_index=["s1", "s2"],
            time_indexes=[[1, 2, 3], [1, 2]],
        )
        mock_predictive_data_indexes(
            monkeypatch,
            cls_name="TemporalTreatmentEffectsTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=["s1", "s2"],
            targets_time_indexes=targets_time_indexes,
            treatments_time_indexes=treatments_time_indexes,
        )

        mock_temporal_treatment_effects_dataset_cls._validate = (  # type: ignore
            dataset.TemporalTreatmentEffectsDataset._validate
        )
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=match_exc):
            mock_temporal_treatment_effects_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
                treatments=data_treatments,
            )


# --- Test DataSet with concrete data (closer to integration tests). ---


@dataclasses.dataclass
class DfsUnderTest:
    df_static: pd.DataFrame
    df_time_series: pd.DataFrame
    df_event: pd.DataFrame


def define_test_dfs() -> DfsUnderTest:
    df_s = pd.DataFrame(
        {
            "sample_idx": ["sample_1", "sample_2", "sample_3"],
            "feat_s_1": [101, 201, 301],
            "feat_s_2": [np.nan, 20.1, 30.1],
            "feat_s_3": ["p", "q", "p"],
        }
    )
    df_s.set_index("sample_idx", drop=True, inplace=True)
    df_s["feat_s_3"] = pd.Categorical(df_s["feat_s_3"])

    df_t = pd.DataFrame(
        {
            "sample_idx": ["sample_1", "sample_1", "sample_1", "sample_1", "sample_2", "sample_2", "sample_3"],
            "time_idx": [
                pd.to_datetime("2000-01-01"),
                pd.to_datetime("2000-01-02"),
                pd.to_datetime("2000-01-03"),
                pd.to_datetime("2000-01-04"),
                pd.to_datetime("2000-02-01"),
                pd.to_datetime("2000-02-02"),
                pd.to_datetime("2000-03-01"),
            ],
            "feat_t_1": [11, 12, 13, 14, 21, 22, 31],
            "feat_t_2": [1.1, np.nan, 1.3, 1.4, np.nan, 2.2, 3.1],
            "feat_t_3": ["a", "a", "b", "a", "a", "b", "b"],
        }
    )
    df_t.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)
    df_t["feat_t_3"] = pd.Categorical(df_t["feat_t_3"])

    df_e = pd.DataFrame(
        {
            "sample_idx": ["sample_1", "sample_2", "sample_3"],
            "feat_e_1": [(5.9, True), (6.1, False), (3.8, True)],
            "feat_e_3": [
                (pd.to_datetime("2000-01-02"), False),
                (pd.to_datetime("2000-01-03"), True),
                (pd.to_datetime("2000-01-01"), True),
            ],
        },
    )
    df_e.set_index("sample_idx", drop=True, inplace=True)

    return DfsUnderTest(df_s, df_t, df_e)


test_dfs = define_test_dfs()


@pytest.fixture
def dummy_dfs_for_split_tests() -> Tuple[pd.DataFrame, ...]:
    size = 100

    df_s = pd.DataFrame(
        {
            "sample_idx": [f"sample_{x}" for x in range(size)],
            "feat_s_1": [x * 0.5 for x in range(size)],
            "feat_s_2": [x * 1.5 for x in range(size)],
        }
    )
    df_s.set_index("sample_idx", drop=True, inplace=True)

    sample_idxs = [[f"sample_{x}"] * 2 for x in range(size)]
    df_t = pd.DataFrame(
        {
            "sample_idx": [item for sublist in sample_idxs for item in sublist],  # Flatten.
            "time_idx": [x * 0.1 for x in range(size * 2)],
            "feat_t_1": [0 for _ in range(size * 2)],
            "feat_t_2": [x * 2.0 + 1.5 for x in range(size * 2)],
        }
    )
    df_t.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

    df_s_target = pd.DataFrame(
        {
            "sample_idx": [f"sample_{x}" for x in range(size)],
            "target": [0] * int(size / 2) + [1] * int(size / 2),
        }
    )
    df_s_target.set_index("sample_idx", drop=True, inplace=True)

    return df_t, df_s, df_s_target


class TestWithConcreteData:
    @pytest.mark.parametrize(
        "time_series, static, target",
        [
            (test_dfs.df_time_series, test_dfs.df_static, test_dfs.df_static.copy()),
            (test_dfs.df_time_series, None, test_dfs.df_static.copy()),
        ],
    )
    def test_init_one_off_prediction_dataset(self, time_series, static, target):
        data = dataset.OneOffPredictionDataset(
            time_series=time_series,
            static=static,
            targets=target,
        )
        assert data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static",
        [
            (test_dfs.df_time_series, test_dfs.df_static),
            (test_dfs.df_time_series, None),
        ],
    )
    def test_init_one_off_prediction_dataset_not_fit_ready(self, time_series, static):
        data = dataset.OneOffPredictionDataset(
            time_series=time_series,
            static=static,
            targets=None,
        )
        assert not data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static, target",
        [
            (
                test_dfs.df_time_series.copy(),
                test_dfs.df_static.copy(),
                test_dfs.df_time_series.copy(),
            ),
            (
                test_dfs.df_time_series.copy(),
                None,
                test_dfs.df_time_series.copy(),
            ),
        ],
    )
    def test_init_temporal_prediction_dataset(self, time_series, static, target):
        data = dataset.TemporalPredictionDataset(
            time_series=time_series,
            static=static,
            targets=target,
        )
        assert data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static",
        [
            (test_dfs.df_time_series.copy(), test_dfs.df_static.copy()),
            (test_dfs.df_time_series.copy(), None),
        ],
    )
    def test_init_temporal_prediction_dataset_not_fit_ready(self, time_series, static):
        data = dataset.TemporalPredictionDataset(
            time_series=time_series,
            static=static,
            targets=None,
        )
        assert not data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static, target",
        [
            (
                test_dfs.df_time_series.copy(),
                test_dfs.df_static.copy(),
                test_dfs.df_event.copy(),
            ),
            (
                test_dfs.df_time_series.copy(),
                None,
                test_dfs.df_event.copy(),
            ),
        ],
    )
    def test_init_time_to_event_analysis_dataset(self, time_series, static, target):
        data = dataset.TimeToEventAnalysisDataset(
            time_series=time_series,
            static=static,
            targets=target,
        )
        assert data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static",
        [
            (test_dfs.df_time_series.copy(), test_dfs.df_static.copy()),
            (test_dfs.df_time_series.copy(), None),
        ],
    )
    def test_init_time_to_event_analysis_dataset_not_fit_ready(self, time_series, static):
        data = dataset.TimeToEventAnalysisDataset(
            time_series=time_series,
            static=static,
            targets=None,
        )
        assert not data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static, target, treatment",
        [
            (
                test_dfs.df_time_series.copy(),
                test_dfs.df_static.copy(),
                test_dfs.df_time_series.copy(),
                test_dfs.df_event.copy(),
            ),
            (
                test_dfs.df_time_series.copy(),
                None,
                test_dfs.df_time_series.copy(),
                test_dfs.df_event.copy(),
            ),
        ],
    )
    def test_init_one_off_treatment_effects_dataset(self, time_series, static, target, treatment):
        data = dataset.OneOffTreatmentEffectsDataset(
            time_series=time_series,
            static=static,
            targets=target,
            treatments=treatment,
        )
        assert data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static, treatment",
        [
            (
                test_dfs.df_time_series.copy(),
                test_dfs.df_static.copy(),
                test_dfs.df_event.copy(),
            ),
            (
                test_dfs.df_time_series.copy(),
                None,
                test_dfs.df_event.copy(),
            ),
        ],
    )
    def test_init_one_off_treatment_effects_dataset_not_fit_ready(self, time_series, static, treatment):
        data = dataset.OneOffTreatmentEffectsDataset(
            time_series=time_series,
            static=static,
            targets=None,
            treatments=treatment,
        )
        assert not data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static",
        [
            (test_dfs.df_time_series.copy(), test_dfs.df_static.copy()),
            (test_dfs.df_time_series.copy(), None),
        ],
    )
    def test_init_one_off_treatment_effects_dataset_fails_validation_no_treatment(self, time_series, static):
        with pytest.raises(ValueError, match=".*treatment.*"):
            dataset.OneOffTreatmentEffectsDataset(
                time_series=time_series,
                static=static,
                targets=None,
                treatments=None,  # type: ignore
            )

    @pytest.mark.parametrize(
        "time_series, static, target, treatment",
        [
            (
                test_dfs.df_time_series.copy(),
                test_dfs.df_static.copy(),
                test_dfs.df_time_series.copy(),
                test_dfs.df_time_series.copy(),
            ),
            (
                test_dfs.df_time_series.copy(),
                None,
                test_dfs.df_time_series.copy(),
                test_dfs.df_time_series.copy(),
            ),
        ],
    )
    def test_init_temporal_treatment_effects_dataset(self, time_series, static, target, treatment):
        data = dataset.TemporalTreatmentEffectsDataset(
            time_series=time_series,
            static=static,
            targets=target,
            treatments=treatment,
        )
        assert data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static, treatment",
        [
            (
                test_dfs.df_time_series.copy(),
                test_dfs.df_static.copy(),
                test_dfs.df_time_series.copy(),
            ),
            (
                test_dfs.df_time_series.copy(),
                None,
                test_dfs.df_time_series.copy(),
            ),
        ],
    )
    def test_init_temporal_treatment_effects_dataset_not_fit_ready(self, time_series, static, treatment):
        data = dataset.TemporalTreatmentEffectsDataset(
            time_series=time_series,
            static=static,
            targets=None,
            treatments=treatment,
        )
        assert not data.fit_ready
        assert data.predict_ready

    @pytest.mark.parametrize(
        "time_series, static",
        [
            (test_dfs.df_time_series.copy(), test_dfs.df_static.copy()),
            (test_dfs.df_time_series.copy(), None),
        ],
    )
    def test_init_temporal_treatment_effects_dataset_fails_validation_no_treatment(self, time_series, static):
        with pytest.raises(ValueError, match=".*treatment.*"):
            dataset.TemporalTreatmentEffectsDataset(
                time_series=time_series,
                static=static,
                targets=None,
                treatments=None,  # type: ignore
            )

    def test_set_properties(self):
        data = dataset.TemporalTreatmentEffectsDataset(
            time_series=test_dfs.df_time_series.copy(),
            static=test_dfs.df_static.copy(),
            targets=test_dfs.df_time_series.copy(),
            treatments=test_dfs.df_time_series.copy(),
        )

        df_s = pd.DataFrame(
            {
                "sample_idx": ["sample_1", "sample_2", "sample_3"],
                "feat_new": [101, 201, 301],
            }
        )
        df_s.set_index("sample_idx", drop=True, inplace=True)

        df_t = pd.DataFrame(
            {
                "sample_idx": ["sample_1", "sample_1", "sample_1", "sample_1", "sample_2", "sample_2", "sample_3"],
                "time_idx": [
                    pd.to_datetime("2000-01-01"),
                    pd.to_datetime("2000-01-02"),
                    pd.to_datetime("2000-01-03"),
                    pd.to_datetime("2000-01-04"),
                    pd.to_datetime("2000-02-01"),
                    pd.to_datetime("2000-02-02"),
                    pd.to_datetime("2000-03-01"),
                ],
                "feat_new": [11, 12, 13, 14, 21, 22, 31],
            }
        )
        df_t.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

        data.static = samples.StaticSamples(df_s)
        data.time_series = samples.TimeSeriesSamples(df_t)
        data.predictive.targets = samples.TimeSeriesSamples(df_t)
        data.predictive.treatments = samples.TimeSeriesSamples(df_t)

        assert data.static.dataframe().equals(df_s)
        assert data.time_series.dataframe().equals(df_t)
        assert data.predictive.targets.dataframe().equals(df_t)
        assert data.predictive.treatments.dataframe().equals(df_t)

    def test_train_test_split(self, dummy_dfs_for_split_tests):
        df_t, df_s, df_s_target = dummy_dfs_for_split_tests
        data = dataset.OneOffPredictionDataset(time_series=df_t, static=df_s, targets=df_s_target)

        data_train, data_test = data.train_test_split(test_size=0.4, random_state=1234)

        assert isinstance(data_train, dataset.OneOffPredictionDataset)
        assert isinstance(data_test, dataset.OneOffPredictionDataset)
        assert len(data_train) + len(data_test) == 100
        assert len(data_test) / 100 == pytest.approx(0.4)

    def test_stratified_kfold_split(self, dummy_dfs_for_split_tests):
        df_t, df_s, df_s_target = dummy_dfs_for_split_tests
        data = dataset.OneOffPredictionDataset(time_series=df_t, static=df_s, targets=df_s_target)
        if TYPE_CHECKING:
            assert data.predictive.targets is not None  # nosec B101
        strat = data.predictive.targets.numpy().reshape((-1,))

        kfold = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)

        n = 0
        for data_train, data_test in data.split(splitter=kfold, y=strat):
            assert isinstance(data_train, dataset.OneOffPredictionDataset)
            assert isinstance(data_test, dataset.OneOffPredictionDataset)
            assert len(data_train) + len(data_test) == 100
            n += 1

        assert n == 5
