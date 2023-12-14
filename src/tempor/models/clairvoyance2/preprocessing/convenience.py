# mypy: ignore-errors

from abc import abstractmethod
from typing import Collection, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..data import Dataset, StaticSamples, TimeSeries, TimeSeriesSamples
from ..data.dataformat import T_FeatureIndexDtype
from ..data.utils import cast_time_series_samples_feature_names_to_str
from ..interface.model import TParams, TransformerModel
from ..interface.requirements import DatasetRequirements, Requirements
from ..utils.common import python_type_from_np_pd_dtype
from ..utils.dev import raise_not_implemented

TFeatureSelector = Union[T_FeatureIndexDtype, Collection[T_FeatureIndexDtype], slice]


class ExtractTC(TransformerModel):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(),
    )

    def _fit(self, data: Dataset, **kwargs) -> "ExtractTC":
        # Nothing happens in `fit` here.
        return self

    @abstractmethod
    def _get_selector_param(self) -> TFeatureSelector:
        ...

    def _extract(
        self,
        temporal_covariates: TimeSeriesSamples,  # type: ignore
    ) -> Tuple[TimeSeriesSamples, Optional[TimeSeriesSamples]]:
        all_features = set(temporal_covariates.feature_names)

        temporal_extracted: TimeSeriesSamples = temporal_covariates[:, self._get_selector_param()]  # type: ignore

        if len(temporal_extracted.features) > 0:
            extracted_features = set(temporal_extracted.feature_names)
            remaining_features = tuple(all_features - extracted_features)

            if len(remaining_features) == 0:
                raise_not_implemented("Selecting all temporal features so that no covariates remain")

            temporal_covariates: TimeSeriesSamples = temporal_covariates[:, remaining_features]  # type: ignore
            # TODO: Need to make sure that __getitem__ supports collection of index items and update typehints.

            temporal_extracted_out = temporal_extracted

        else:
            temporal_extracted_out = None

        return temporal_covariates, temporal_extracted_out


class _ExtractTargetsTCParams(NamedTuple):
    targets: TFeatureSelector = tuple()


class TemporalTargetsExtractor(ExtractTC):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(),
    )
    DEFAULT_PARAMS: _ExtractTargetsTCParams = _ExtractTargetsTCParams()

    def _get_selector_param(self) -> TFeatureSelector:
        return self.params.targets

    def _transform(self, data: Dataset, **kwargs) -> Dataset:
        data = data.copy()

        temporal_covariates, temporal_targets = self._extract(data.temporal_covariates)
        data.temporal_covariates = temporal_covariates
        data.temporal_targets = temporal_targets

        return data


class _ExtractTreatmentsTCParams(NamedTuple):
    treatments: TFeatureSelector = tuple()


class TemporalTreatmentsExtractor(ExtractTC):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(),
    )
    DEFAULT_PARAMS: _ExtractTreatmentsTCParams = _ExtractTreatmentsTCParams()

    def _get_selector_param(self) -> TFeatureSelector:
        return self.params.treatments

    def _transform(self, data: Dataset, **kwargs) -> Dataset:
        data = data.copy()

        temporal_covariates, temporal_treatments = self._extract(data.temporal_covariates)
        data.temporal_covariates = temporal_covariates
        data.temporal_treatments = temporal_treatments

        return data


class _AddTimeIndexFeatureTCParams(NamedTuple):
    add_time_index: bool = False
    add_time_delta: bool = True
    time_delta_pad_at_back: bool = False
    time_delta_pad_value: float = 0.0


class TimeIndexFeatureConcatenator(TransformerModel):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(
            requires_all_temporal_data_index_numeric=True,
        ),
    )
    DEFAULT_PARAMS: _AddTimeIndexFeatureTCParams = _AddTimeIndexFeatureTCParams()

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(params)
        if self.params.add_time_index is False and self.params.add_time_delta is False:
            raise ValueError("Must set at least one of `add_time_index` or `add_time_delta` to True")

    def _fit(self, data: Dataset, **kwargs) -> "TimeIndexFeatureConcatenator":
        # Nothing happens in `fit` here.
        return self

    def _transform(self, data: Dataset, **kwargs) -> Dataset:
        data = data.copy()

        cast_time_series_samples_feature_names_to_str(data.temporal_covariates)
        # ^ Since we are adding features by string names below.

        for ts in data.temporal_covariates:
            df = ts.df
            df_new = df.copy()

            if self.params.add_time_delta:
                diff = np.diff(df.index.values)
                if self.params.time_delta_pad_at_back is False:
                    diff = np.append(self.params.time_delta_pad_value, diff)
                else:
                    diff = np.append(diff, self.params.time_delta_pad_value)
                df_new.insert(0, "time_delta", diff)

            if self.params.add_time_index:
                df_new.insert(0, "time_index", df.index)

            ts.df = df_new
            # NOTE: No change to feature categorical definitions, as these new features are all numeric.

        return data


class _AddStaticCovariatesTCParams(NamedTuple):
    feature_name_prefix: Optional[str] = "static"
    append_at_beginning: bool = False
    drop_static_covariates: bool = False


class StaticFeaturesConcatenator(TransformerModel):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(
            requires_static_covariates_present=True,
        )
    )
    DEFAULT_PARAMS: _AddStaticCovariatesTCParams = _AddStaticCovariatesTCParams()

    def _fit(self, data: Dataset, **kwargs) -> "StaticFeaturesConcatenator":
        # Nothing happens in `fit` here.
        return self

    def _transform(self, data: Dataset, **kwargs) -> Dataset:
        data = data.copy()
        assert data.static_covariates is not None

        if (
            self.params.feature_name_prefix is not None
            or python_type_from_np_pd_dtype(data.static_covariates.df.columns.dtype) == object  # type: ignore
        ):
            # If static covariate features are indexed by str, do the same for temporal covariates.
            cast_time_series_samples_feature_names_to_str(data.temporal_covariates)

        s_cov: StaticSamples = data.static_covariates
        s_cov_new_feature_names: Dict[T_FeatureIndexDtype, T_FeatureIndexDtype] = {
            k: (f"{self.params.feature_name_prefix}_{k}" if self.params.feature_name_prefix is not None else k)
            for k in s_cov.feature_names
        }
        clashing_feature_names = set(s_cov_new_feature_names.values()).intersection(
            data.temporal_covariates.feature_names
        )
        if len(clashing_feature_names) > 0:
            raise ValueError(
                f"Features named {clashing_feature_names} clash with existing temporal covariate features. "
                "Try setting/changing `feature_name_prefix` parameter."
            )

        t_cov_sample: TimeSeries
        for sample_idx, t_cov_sample in zip(data.temporal_covariates.sample_indices, data.temporal_covariates):
            t_cov_sample_df = t_cov_sample.df
            t_cov_sample_df_new = t_cov_sample_df.copy()

            # Get sample static covariates:
            s_cov_sample: pd.DataFrame = data.static_covariates.df.loc[[sample_idx], :]
            # Repeat this as long as the time index on the temporal covariates.
            to_append = pd.concat([s_cov_sample] * t_cov_sample_df.shape[0], ignore_index=True)
            # Give this new feature names as columns.
            to_append.rename(mapper=s_cov_new_feature_names, axis=1, inplace=True)
            # Give this the same exact index as temporal features dataframe.
            to_append.set_index(t_cov_sample_df.index, drop=True, inplace=True)

            # Actually append the features.
            if self.params.append_at_beginning is False:
                t_cov_sample_df_new = pd.concat([t_cov_sample_df_new, to_append], axis=1)
            else:
                t_cov_sample_df_new = pd.concat([to_append, t_cov_sample_df_new], axis=1)
            assert (t_cov_sample_df.index == t_cov_sample_df_new.index).all()

            # print(t_cov_sample_df_new)
            t_cov_sample.df = t_cov_sample_df_new
            # t_cov_sample.df = t_cov_sample_df_new

        if self.params.drop_static_covariates:
            data.static_covariates = None

        return data
