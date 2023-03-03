# pylint: disable=redefined-outer-name, unused-argument, protected-access

import dataclasses
from typing import Type
from unittest.mock import Mock

import pytest

from tempor.data import dataset, predictive, samples


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


def mock_sample_index(monkeypatch, cls_name, sample_index):
    instance = Mock()
    instance.sample_index = Mock(return_value=sample_index)
    monkeypatch.setattr(samples, cls_name, Mock(return_value=instance))


def mock_predictive_data_sample_index(monkeypatch, cls_name, targets_sample_index, treatments_sample_index):
    targets_instance = Mock()
    targets_instance.sample_index = Mock(return_value=targets_sample_index)
    treatments_instance = Mock()
    treatments_instance.sample_index = Mock(return_value=treatments_sample_index)
    MockOneOffPredictionTaskData = Mock(return_value=Mock(targets=targets_instance, treatments=treatments_instance))
    monkeypatch.setattr(predictive, cls_name, MockOneOffPredictionTaskData)


# --- Dataset base class. ---


class MockedDataset(dataset.Dataset):
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
    MockedDataset.mock_validate_call = Mock()
    MockedDataset.mock_init_predictive = Mock()
    return MockedDataset


class TestDataset:
    def test_init(self, mock_dataset_cls: Type[MockedDataset], mocked_depends: MockedDepends):
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

    def test_init_no_targets_provided(self, mock_dataset_cls: Type[MockedDataset], mocked_depends: MockedDepends):
        data_time_series = Mock()
        data_static = Mock()
        data_targets = None

        dummy_dataset = mock_dataset_cls(
            time_series=data_time_series,
            static=data_static,
            targets=data_targets,
        )

        mocked_depends.TimeSeriesSamples.assert_called_once_with(data_time_series)
        mocked_depends.StaticSamples.assert_called_once_with(data_static)
        dummy_dataset.mock_init_predictive.assert_not_called()
        dummy_dataset.mock_validate_call.assert_called_once()

    @pytest.mark.parametrize(
        "static, expected",
        [
            (Mock(), True),
            (None, False),
        ],
    )
    def test_has_static(
        self, static, expected: bool, mock_dataset_cls: Type[MockedDataset], mocked_depends: MockedDepends
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
            (None, False, None),
        ],
    )
    def test_has_predictive_data_and_predictive_task(
        self,
        targets,
        expected_has_predictive_data: bool,
        expected_predictive_task,
        mock_dataset_cls: Type[MockedDataset],
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
    def test_validate_passes(self, static, mock_dataset_cls: Type[MockedDataset], monkeypatch):
        data_time_series = Mock()
        mock_sample_index(monkeypatch, cls_name="StaticSamples", sample_index=["s1", "s2"])
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])

        mock_dataset_cls.validate = dataset.Dataset.validate  # type: ignore  # Enable validation code.

        dummy_dataset = mock_dataset_cls(
            time_series=data_time_series,
            static=static,
            targets=None,
        )

        dummy_dataset.mock_validate_call.assert_called_once()

    def test_validate_fails_sample_index_mismatch(self, mock_dataset_cls: Type[MockedDataset], monkeypatch):
        data_time_series = Mock()
        data_static = Mock()
        mock_sample_index(monkeypatch, cls_name="StaticSamples", sample_index=["s1", "s2"])
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2", "s3"])

        mock_dataset_cls.validate = dataset.Dataset.validate  # type: ignore  # Enable validation code.

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
        mocked_depends.OneOffPredictionTaskData.assert_called_once_with(targets=data_targets, dummy="kwarg")
        dummy_dataset.mock_validate_call.assert_called_once()

    def test_validate_passes(
        self, mock_one_off_prediction_dataset_cls: Type[MockedOneOffPredictionDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
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
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
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
        mocked_depends.TemporalPredictionTaskData.assert_called_once_with(targets=data_targets, dummy="kwarg")
        dummy_dataset.mock_validate_call.assert_called_once()

    def test_validate_passes(
        self, mock_temporal_prediction_dataset_cls: Type[MockedTemporalPredictionDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
            monkeypatch,
            cls_name="TemporalPredictionTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=None,
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
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
            monkeypatch,
            cls_name="TemporalPredictionTaskData",
            targets_sample_index=["s1", "s2", "s3"],
            treatments_sample_index=None,
        )

        mock_temporal_prediction_dataset_cls._validate = dataset.TemporalPredictionDataset._validate  # type: ignore
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=".*sample_index.*targets.*time series.*"):
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
        mocked_depends.TimeToEventAnalysisTaskData.assert_called_once_with(targets=data_targets, dummy="kwarg")
        dummy_dataset.mock_validate_call.assert_called_once()

    def test_validate_passes(
        self, mock_time_to_event_analysis_dataset_cls: Type[MockedTimeToEventAnalysisDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
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
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
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
            targets=data_targets, treatments=data_treatments, dummy="kwarg"
        )
        dummy_dataset.mock_validate_call.assert_called_once()

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
                treatments=None,
            )

    def test_validate_passes(
        self, mock_one_off_treatment_effects_dataset_cls: Type[MockedOneOffTreatmentEffectsDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        data_treatments = Mock()
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
            monkeypatch,
            cls_name="OneOffTreatmentEffectsTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=["s1", "s2"],
        )

        mock_one_off_treatment_effects_dataset_cls._validate = dataset.OneOffTreatmentEffectsDataset._validate  # type: ignore
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
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
            monkeypatch,
            cls_name="OneOffTreatmentEffectsTaskData",
            targets_sample_index=targets_sample_index,
            treatments_sample_index=treatments_sample_index,
        )

        mock_one_off_treatment_effects_dataset_cls._validate = dataset.OneOffTreatmentEffectsDataset._validate  # type: ignore
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=match_exc):
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
            targets=data_targets, treatments=data_treatments, dummy="kwarg"
        )
        dummy_dataset.mock_validate_call.assert_called_once()

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
                treatments=None,
            )

    def test_validate_passes(
        self, mock_temporal_treatment_effects_dataset_cls: Type[MockedTemporalTreatmentEffectsDataset], monkeypatch
    ):
        data_time_series = Mock()
        data_static = None
        data_targets = Mock()
        data_treatments = Mock()
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
            monkeypatch,
            cls_name="TemporalTreatmentEffectsTaskData",
            targets_sample_index=["s1", "s2"],
            treatments_sample_index=["s1", "s2"],
        )

        mock_temporal_treatment_effects_dataset_cls._validate = dataset.TemporalTreatmentEffectsDataset._validate  # type: ignore
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
        mock_sample_index(monkeypatch, cls_name="TimeSeriesSamples", sample_index=["s1", "s2"])
        mock_predictive_data_sample_index(
            monkeypatch,
            cls_name="TemporalTreatmentEffectsTaskData",
            targets_sample_index=targets_sample_index,
            treatments_sample_index=treatments_sample_index,
        )

        mock_temporal_treatment_effects_dataset_cls._validate = dataset.TemporalTreatmentEffectsDataset._validate  # type: ignore
        # ^ Enable validation code.

        with pytest.raises(ValueError, match=match_exc):
            mock_temporal_treatment_effects_dataset_cls(
                time_series=data_time_series,
                static=data_static,
                targets=data_targets,
                treatments=data_treatments,
            )


# TODO: May want to add some tests with actual data frame / numpy array inputs.
