# mypy: ignore-errors

import math
from typing import Any, Optional, Sequence

import numpy as np

from ..data import Dataset, StaticSamples, TimeSeriesSamples


class DummyDatasetGenerator:
    def __init__(
        self,
        n_samples: int,
        temporal_covariates_n_features: int,
        temporal_covariates_max_len: int,
        temporal_covariates_missing_prob: float,
        static_covariates_n_features: int,
        static_covariates_missing_prob: float,
        temporal_targets_n_features: int,
        temporal_targets_n_categories: Optional[int],
        temporal_treatments_n_features: int,
        temporal_treatments_n_categories: Optional[int],
    ) -> None:
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

    @staticmethod
    def _random_lengths(rng, max_len, n_samples) -> np.ndarray:
        return rng.integers(low=math.ceil(max_len / 2), high=max_len, size=(n_samples,)) + 1

    @staticmethod
    def _apply_missing(array: np.ndarray, rng, missing_prob: float) -> np.ndarray:
        array = array.copy()
        missing = rng.uniform(low=0.0, high=1.0, size=array.shape) < missing_prob
        array[missing] = np.nan
        return array

    @staticmethod
    def _generate_temporal_continuous(
        random_seed: int,
        max_len: int,
        n_samples: int,
        n_features: int,
        missing_prob: float,
        lengths: Optional[Sequence[int]] = None,
    ) -> TimeSeriesSamples:
        rng = np.random.default_rng(seed=random_seed)
        lens: Any = DummyDatasetGenerator._random_lengths(rng, max_len, n_samples) if lengths is None else lengths
        noise_mus = rng.standard_normal(size=(n_features,))
        noise_sigmas = 1.0 + rng.standard_normal(size=(n_features,))
        list_ = []
        for sample_len in lens:
            trend = np.tile(np.arange(sample_len), (n_features, 1)).T
            noise = noise_mus + noise_sigmas * rng.standard_normal(size=(sample_len, n_features))
            final = trend + noise
            final = DummyDatasetGenerator._apply_missing(final, rng, missing_prob)
            list_.append(final)
        return TimeSeriesSamples(data=list_, sample_indices=None)

    @staticmethod
    def _generate_temporal_categorical(
        random_seed: int,
        max_len: int,
        n_samples: int,
        n_features: int,
        n_categories: int,
        lengths: Optional[Sequence[int]] = None,
    ) -> TimeSeriesSamples:
        rng = np.random.default_rng(seed=random_seed)
        lens: Any = DummyDatasetGenerator._random_lengths(rng, max_len, n_samples) if lengths is None else lengths
        list_ = []
        for sample_len in lens:
            array = rng.integers(low=0, high=n_categories, size=(sample_len, n_features))
            list_.append(array)
        return TimeSeriesSamples(data=list_, sample_indices=None)

    @staticmethod
    def _generate_static_continuous(
        random_seed: int,
        n_samples: int,
        n_features: int,
        missing_prob: float,
    ) -> StaticSamples:
        rng = np.random.default_rng(seed=random_seed)
        mus = 2.0 + rng.standard_normal(size=(n_features,))
        sigmas = 0.5 * rng.standard_normal(size=(n_features,))
        array = mus + sigmas * rng.standard_normal(size=(n_samples, n_features))
        array = DummyDatasetGenerator._apply_missing(array, rng, missing_prob)
        return StaticSamples(data=array)

    def generate(self, random_seed: int) -> Dataset:
        temporal_covariates = self._generate_temporal_continuous(
            random_seed,
            max_len=self.temporal_covariates_max_len,
            n_samples=self.n_samples,
            n_features=self.temporal_covariates_n_features,
            missing_prob=self.temporal_covariates_missing_prob,
        )

        if self.static_covariates_n_features > 0:
            static_covariates = self._generate_static_continuous(
                random_seed,
                n_samples=self.n_samples,
                n_features=self.static_covariates_n_features,
                missing_prob=self.static_covariates_missing_prob,
            )
        else:
            static_covariates = None

        if self.temporal_targets_n_features > 0:
            if self.temporal_targets_n_categories is not None:
                temporal_targets = self._generate_temporal_categorical(
                    random_seed,
                    max_len=-1,
                    n_samples=self.n_samples,
                    n_features=self.temporal_targets_n_features,
                    n_categories=self.temporal_targets_n_categories,
                    lengths=temporal_covariates.n_timesteps_per_sample,
                )
            else:
                temporal_targets = self._generate_temporal_continuous(
                    random_seed,
                    max_len=-1,
                    n_samples=self.n_samples,
                    n_features=self.temporal_targets_n_features,
                    missing_prob=0.0,
                    lengths=temporal_covariates.n_timesteps_per_sample,
                )
        else:
            temporal_targets = None

        if self.temporal_treatments_n_features > 0:
            if self.temporal_treatments_n_categories is not None:
                temporal_treatments = self._generate_temporal_categorical(
                    random_seed,
                    max_len=-1,
                    n_samples=self.n_samples,
                    n_features=self.temporal_treatments_n_features,
                    n_categories=self.temporal_treatments_n_categories,
                    lengths=temporal_covariates.n_timesteps_per_sample,
                )
            else:
                temporal_treatments = self._generate_temporal_continuous(
                    random_seed,
                    max_len=-1,
                    n_samples=self.n_samples,
                    n_features=self.temporal_treatments_n_features,
                    missing_prob=0.0,
                    lengths=temporal_covariates.n_timesteps_per_sample,
                )
        else:
            temporal_treatments = None

        return Dataset(
            temporal_covariates=temporal_covariates,
            static_covariates=static_covariates,
            temporal_targets=temporal_targets,
            temporal_treatments=temporal_treatments,
        )


def dummy_dataset(
    n_samples: int = 100,
    temporal_covariates_n_features: int = 5,
    temporal_covariates_max_len: int = 20,
    temporal_covariates_missing_prob: float = 0.1,
    static_covariates_n_features: int = 0,
    static_covariates_missing_prob: float = 0.0,
    temporal_targets_n_features: int = 0,
    temporal_targets_n_categories: Optional[int] = None,
    temporal_treatments_n_features: int = 0,
    temporal_treatments_n_categories: Optional[int] = None,
    random_seed: int = 12345,
) -> Dataset:
    dummy_dataset_generator = DummyDatasetGenerator(
        n_samples=n_samples,
        temporal_covariates_n_features=temporal_covariates_n_features,
        temporal_covariates_max_len=temporal_covariates_max_len,
        temporal_covariates_missing_prob=temporal_covariates_missing_prob,
        static_covariates_n_features=static_covariates_n_features,
        static_covariates_missing_prob=static_covariates_missing_prob,
        temporal_targets_n_features=temporal_targets_n_features,
        temporal_targets_n_categories=temporal_targets_n_categories,
        temporal_treatments_n_features=temporal_treatments_n_features,
        temporal_treatments_n_categories=temporal_treatments_n_categories,
    )
    return dummy_dataset_generator.generate(random_seed=random_seed)
