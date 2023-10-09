from typing import Optional, cast

from clairvoyance2.datasets.dummy import dummy_dataset

from tempor.data import dataset
from tempor.data.clv2conv import clairvoyance2_dataset_to_tempor_dataset
from tempor.data.datasources import datasource


# TODO: Docstring.
class DummyTemporalPredictionDataSource(datasource.TemporalPredictionDataSource):
    def __init__(
        self,
        n_samples: int = 100,
        temporal_covariates_n_features: int = 5,
        temporal_covariates_max_len: int = 20,
        temporal_covariates_missing_prob: float = 0.1,
        static_covariates_n_features: int = 3,
        static_covariates_missing_prob: float = 0.0,
        temporal_targets_n_features: int = 2,
        temporal_targets_n_categories: Optional[int] = None,
        random_state: int = 12345,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.n_samples = n_samples
        self.temporal_covariates_n_features = temporal_covariates_n_features
        self.temporal_covariates_max_len = temporal_covariates_max_len
        self.temporal_covariates_missing_prob = temporal_covariates_missing_prob
        self.static_covariates_n_features = static_covariates_n_features
        self.static_covariates_missing_prob = static_covariates_missing_prob
        self.temporal_targets_n_features = temporal_targets_n_features
        self.temporal_targets_n_categories = temporal_targets_n_categories
        self.random_state = random_state

    @staticmethod
    def url() -> None:
        return None

    @staticmethod
    def dataset_dir() -> None:
        return None

    def load(self, **kwargs) -> dataset.TemporalPredictionDataset:
        clv_dataset = dummy_dataset(
            n_samples=self.n_samples,
            temporal_covariates_n_features=self.temporal_covariates_n_features,
            temporal_covariates_max_len=self.temporal_covariates_max_len,
            temporal_covariates_missing_prob=self.temporal_covariates_missing_prob,
            static_covariates_n_features=self.static_covariates_n_features,
            temporal_targets_n_features=self.temporal_targets_n_features,
            temporal_targets_n_categories=self.temporal_targets_n_categories,
            random_seed=self.random_state,
            # No temporal treatments features in this case:
            temporal_treatments_n_features=0,
            temporal_treatments_n_categories=None,
        )
        data = clairvoyance2_dataset_to_tempor_dataset(clv_dataset)
        data = cast(dataset.TemporalPredictionDataset, data)
        return data


# TODO: Docstring.
class DummyTemporalTreatmentEffectsDataSource(datasource.TemporalTreatmentEffectsDataSource):
    def __init__(
        self,
        n_samples: int = 100,
        temporal_covariates_n_features: int = 5,
        temporal_covariates_max_len: int = 20,
        temporal_covariates_missing_prob: float = 0.1,
        static_covariates_n_features: int = 3,
        static_covariates_missing_prob: float = 0.0,
        temporal_targets_n_features: int = 2,
        temporal_targets_n_categories: Optional[int] = None,
        temporal_treatments_n_features: int = 2,
        temporal_treatments_n_categories: Optional[int] = None,
        random_state: int = 12345,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.n_samples = n_samples
        self.temporal_covariates_n_features = temporal_covariates_n_features
        self.temporal_covariates_max_len = temporal_covariates_max_len
        self.temporal_covariates_missing_prob = temporal_covariates_missing_prob
        self.static_covariates_n_features = static_covariates_n_features
        self.static_covariates_missing_prob = static_covariates_missing_prob
        self.temporal_targets_n_features = temporal_targets_n_features
        self.temporal_targets_n_categories = temporal_targets_n_categories
        self.temporal_treatments_n_features = temporal_treatments_n_features
        self.temporal_treatments_n_categories = temporal_treatments_n_categories
        self.random_state = random_state

    @staticmethod
    def url() -> None:
        return None

    @staticmethod
    def dataset_dir() -> None:
        return None

    def load(self, **kwargs) -> dataset.TemporalTreatmentEffectsDataset:
        clv_dataset = dummy_dataset(
            n_samples=self.n_samples,
            temporal_covariates_n_features=self.temporal_covariates_n_features,
            temporal_covariates_max_len=self.temporal_covariates_max_len,
            temporal_covariates_missing_prob=self.temporal_covariates_missing_prob,
            static_covariates_n_features=self.static_covariates_n_features,
            temporal_targets_n_features=self.temporal_targets_n_features,
            temporal_targets_n_categories=self.temporal_targets_n_categories,
            random_seed=self.random_state,
            # There are treatments features in this case:
            temporal_treatments_n_features=self.temporal_treatments_n_features,
            temporal_treatments_n_categories=self.temporal_treatments_n_categories,
        )
        data = clairvoyance2_dataset_to_tempor_dataset(clv_dataset)
        data = cast(dataset.TemporalTreatmentEffectsDataset, data)
        return data
