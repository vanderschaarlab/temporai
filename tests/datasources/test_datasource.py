from tempor.data import data_typing, dataset
from tempor.datasources import datasource


def test_datasources():
    class MyOneOffPredictionDataSource(datasource.OneOffPredictionDataSource):
        name = "my_one_off_prediction_datasource"
        category = "dummy_category"
        plugin_type = "datasource"

        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    class MyTemporalPredictionDataSource(datasource.TemporalPredictionDataSource):
        name = "my_temporal_prediction_datasource"
        category = "dummy_category"
        plugin_type = "datasource"

        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    class MyTimeToEventAnalysisDataSource(datasource.TimeToEventAnalysisDataSource):
        name = "my_time_to_event_analysis_datasource"
        category = "dummy_category"
        plugin_type = "datasource"

        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    class MyOneOffTreatmentEffectsDataSource(datasource.OneOffTreatmentEffectsDataSource):
        name = "my_one_off_treatment_effects_datasource"
        category = "dummy_category"
        plugin_type = "datasource"

        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url() -> None:
            return None

    class MyTemporalTreatmentEffectsDataSource(datasource.TemporalTreatmentEffectsDataSource):
        name = "my_temporal_treatment_effects_datasource"
        category = "dummy_category"
        plugin_type = "datasource"

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
        name = "dummy_name"
        category = "dummy_category"
        plugin_type = "datasource"

        def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
            raise NotImplementedError

        @staticmethod
        def dataset_dir() -> None:
            return None

        @staticmethod
        def url():
            return "some_url"

    class CaseNoUrl(datasource.OneOffPredictionDataSource):
        name = "dummy_name"
        category = "dummy_category"
        plugin_type = "datasource"

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
