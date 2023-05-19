# pylint: disable=redefined-outer-name,unused-argument

from unittest.mock import Mock

import pytest

from tempor.data import data_typing, predictive


@pytest.fixture
def mock_samples(monkeypatch):
    from tempor.data import samples

    monkeypatch.setattr(samples, "StaticSamples", Mock())
    monkeypatch.setattr(samples, "EventSamples", Mock())
    monkeypatch.setattr(samples, "TimeSeriesSamples", Mock())


def test_predictive(mock_samples):
    class MyOneOffPredictionTaskData(predictive.OneOffPredictionTaskData):
        pass

    class MyTemporalPredictionTaskData(predictive.TemporalPredictionTaskData):
        pass

    class MyTimeToEventAnalysisTaskData(predictive.TimeToEventAnalysisTaskData):
        pass

    class MyOneOffTreatmentEffectsTaskData(predictive.OneOffTreatmentEffectsTaskData):
        pass

    class MyTemporalTreatmentEffectsTaskData(predictive.TemporalTreatmentEffectsTaskData):
        pass

    p1 = MyOneOffPredictionTaskData(parent_dataset=Mock(), targets=None)
    p2 = MyTemporalPredictionTaskData(parent_dataset=Mock(), targets=None)
    p3 = MyTimeToEventAnalysisTaskData(parent_dataset=Mock(), targets=None)
    p4 = MyOneOffTreatmentEffectsTaskData(parent_dataset=Mock(), targets=None, treatments=None)  # type: ignore
    p5 = MyTemporalTreatmentEffectsTaskData(parent_dataset=Mock(), targets=None, treatments=None)  # type: ignore

    assert p1.predictive_task == data_typing.PredictiveTask.ONE_OFF_PREDICTION
    assert p2.predictive_task == data_typing.PredictiveTask.TEMPORAL_PREDICTION
    assert p3.predictive_task == data_typing.PredictiveTask.TIME_TO_EVENT_ANALYSIS
    assert p4.predictive_task == data_typing.PredictiveTask.ONE_OFF_TREATMENT_EFFECTS
    assert p5.predictive_task == data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS


def test_repr(mock_samples):
    class MyOneOffPredictionTaskData(predictive.OneOffPredictionTaskData):
        pass

    p = MyOneOffPredictionTaskData(parent_dataset=Mock(), targets=None)
    assert "targets=None" in str(p)
    assert "treatments" not in str(p)

    p = MyOneOffPredictionTaskData(parent_dataset=Mock(), targets=Mock())
    assert "targets=" in str(p) and "targets=None" not in str(p)
    assert "treatments" not in str(p)

    class MyTemporalTreatmentEffectsTaskData(predictive.TemporalTreatmentEffectsTaskData):
        pass

    p = MyTemporalTreatmentEffectsTaskData(parent_dataset=Mock(), targets=Mock(), treatments=Mock())
    assert "targets=" in str(p) and "targets=None" not in str(p)
    assert "treatments=" in str(p) and "treatments=None" not in str(p)
