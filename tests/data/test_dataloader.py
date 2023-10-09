from tempor.data import data_typing, dataset
from tempor.data.datasources import datasource


def test_dataloaders():
    class MyOneOffPredictionDataSource(datasource.OneOffPredictionDataSource):
        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    class MyTemporalPredictionDataSource(datasource.TemporalPredictionDataSource):
        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    class MyTimeToEventAnalysisDataSource(datasource.TimeToEventAnalysisDataSource):
        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    class MyOneOffTreatmentEffectsDataSource(datasource.OneOffTreatmentEffectsDataSource):
        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    class MyTemporalTreatmentEffectsDataSource(datasource.TemporalTreatmentEffectsDataSource):
        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    dl_oneoff_pred = MyOneOffPredictionDataSource()
    dl_temporal_pred = MyTemporalPredictionDataSource()
    dl_tte = MyTimeToEventAnalysisDataSource()
    dl_oneoff_tr = MyOneOffTreatmentEffectsDataSource()
    dl_temporal_tr = MyTemporalTreatmentEffectsDataSource()

    assert dl_oneoff_pred.dataset_dir() is None
    assert dl_oneoff_pred.url() is None
    assert dl_oneoff_pred.predictive_task == data_typing.PredictiveTask.ONE_OFF_PREDICTION

    assert dl_temporal_pred.dataset_dir() is None
    assert dl_temporal_pred.url() is None
    assert dl_temporal_pred.predictive_task == data_typing.PredictiveTask.TEMPORAL_PREDICTION

    assert dl_tte.dataset_dir() is None
    assert dl_tte.url() is None
    assert dl_tte.predictive_task == data_typing.PredictiveTask.TIME_TO_EVENT_ANALYSIS

    assert dl_oneoff_tr.dataset_dir() is None
    assert dl_oneoff_tr.url() is None
    assert dl_oneoff_tr.predictive_task == data_typing.PredictiveTask.ONE_OFF_TREATMENT_EFFECTS

    assert dl_temporal_tr.dataset_dir() is None
    assert dl_temporal_tr.url() is None
    assert dl_temporal_tr.predictive_task == data_typing.PredictiveTask.TEMPORAL_TREATMENT_EFFECTS


def test_requires_internet():
    class CaseUrl(datasource.OneOffPredictionDataSource):
        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url():
            return "some_url"

    class CaseNoUrl(datasource.OneOffPredictionDataSource):
        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url():
            return None

    case_url = CaseUrl()
    case_no_url = CaseNoUrl()

    assert case_url.requires_internet()
    assert not case_no_url.requires_internet()
