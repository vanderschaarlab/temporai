# mypy: ignore-errors

# NOTE:
# Experimental, minimally tested, will be significantly changed and improved.

from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from ..components.torch import interfaces as ti
from ..components.torch import synctwin_train_utils
from ..components.torch.synctwin_models import LinearDecoder, RegularDecoder, RegularEncoder, SyncTwinModule
from ..data import Dataset, TimeSeries, TimeSeriesSamples
from ..data.constants import T_SamplesIndexDtype
from ..data.utils import time_index_equal, time_index_utils, to_counterfactual_predictions
from ..interface import (
    Horizon,
    TCounterfactualPredictions,
    TDefaultParams,
    TimeIndexHorizon,
    TParams,
    TPredictOutput,
    TreatmentEffectsModel,
    TTreatmentScenarios,
)
from ..interface import requirements as r
from ..utils.array_manipulation import compute_deltas
from ..utils.dev import NEEDED


class _DefaultParams(NamedTuple):
    # Main hyperparameters:
    hidden_size: int = 20
    tau: float = 1.0
    lambda_prognostic: float = 1.0
    lambda_reconstruction: float = 1.0
    batch_size: int = 32
    pretraining_iterations: int = 5_000
    matching_iterations: int = 20_000
    inference_iterations: int = 20_000
    # Misc:
    use_validation_set_in_training: bool = True
    treatment_status_is_treated: int = 1


class SyncTwinTensors(NamedTuple):
    x_full: torch.Tensor
    t_full: torch.Tensor
    mask_full: torch.Tensor
    batch_ind_full: torch.Tensor
    y_full: torch.Tensor
    y_control: torch.Tensor
    y_mask_full: torch.Tensor


# TODO: When returning the result array, MUST ASSIGN TO APPROPRIATE SAMPLE INDICES!
class SyncTwinRegressor(
    TreatmentEffectsModel, ti.OrganizedTreatmentEffectsModuleMixin, ti.OrganizedPredictorModuleMixin, ti.OrganizedModule
):
    requirements: r.Requirements = r.Requirements(
        dataset_requirements=r.DatasetRequirements(
            temporal_covariates_value_type=r.DataValueOpts.NUMERIC,
            temporal_targets_value_type=r.DataValueOpts.NUMERIC,
            static_covariates_value_type=r.DataValueOpts.NUMERIC,
            event_treatments_value_type=r.DataValueOpts.NUMERIC_BINARY,
            requires_no_missing_data=True,  # TODO: For now, eventually allow.
        ),
        prediction_requirements=r.PredictionRequirements(
            target_data_structure=r.DataStructureOpts.TIME_SERIES,
            horizon_type=r.HorizonOpts.TIME_INDEX,
        ),
        treatment_effects_requirements=r.TreatmentEffectsRequirements(
            treatment_data_structure=r.DataStructureOpts.EVENT,
        ),
    )
    DEFAULT_PARAMS: TDefaultParams = _DefaultParams()

    # Fixed params:
    _lambda_express: float = 1.0
    _reg_B: float = 0.0
    _self_expressive_lr: float = 0.001
    _validation_set_frac = 0.5
    _prediction_compute_iters = 500

    expected_treatment_statuses = (0, 1)

    def _get_other_treatment_status(self, treatment_status_indicator) -> int:
        assert treatment_status_indicator in self.expected_treatment_statuses
        return int(1 - treatment_status_indicator)

    def __init__(self, params: Optional[TParams] = None) -> None:
        TreatmentEffectsModel.__init__(self, params)
        ti.OrganizedModule.__init__(self)

        # Quick validation:
        if self.params.treatment_status_is_treated not in self.expected_treatment_statuses:
            raise ValueError(
                f"`treatment_status_is_treated` must be one of: {self.expected_treatment_statuses} "
                f"but was {self.params.treatment_status_is_treated}"
            )
        self._treated_indicator = self.params.treatment_status_is_treated
        self._control_indicator = self._get_other_treatment_status(self.params.treatment_status_is_treated)

        # Components:
        self.encoder: Optional[nn.Module] = NEEDED
        self.decoder: Optional[nn.Module] = NEEDED
        self.decoder_y: Optional[nn.Module] = NEEDED
        self.synctwin: Optional[SyncTwinModule] = NEEDED

        # Helpers:
        self._predict_synctwin_n_unit = None
        self._predict_synctwin_n_treated = None
        self._pretraining_test_freq = max(self.params.pretraining_iterations // 10, 1)
        self._matching_test_freq = max(self.params.matching_iterations // 10, 1)
        self._inference_iterations_test_freq = max(self.params.inference_iterations // 10, 1)

    @property
    def treated_indicator(self) -> int:
        return self._treated_indicator

    @property
    def control_indicator(self) -> int:
        return self._control_indicator

    @staticmethod
    def _extract_pre_treatment(
        time_series_samples: TimeSeriesSamples, event_time_indexes, name: str
    ) -> TimeSeriesSamples:
        time_series_samples = time_index_utils.time_series_samples.take_all_before_start(  # type: ignore
            time_series_samples, event_time_indexes
        )
        if not time_series_samples.all_samples_same_n_timesteps:
            raise RuntimeError(
                f"{SyncTwinRegressor.__name__} requires that {name} up to the treatment event time "
                "have the same number of timesteps but this was not the case"
            )
        return time_series_samples

    @staticmethod
    def _extract_post_treatment(
        time_series_samples: TimeSeriesSamples, event_time_indexes, name: str
    ) -> TimeSeriesSamples:
        time_series_samples = time_index_utils.time_series_samples.take_all_from_start(  # type: ignore
            time_series_samples, event_time_indexes
        )
        if not time_series_samples.all_samples_same_n_timesteps:
            raise RuntimeError(
                f"{SyncTwinRegressor.__name__} requires that {name} from treatment event time on "
                "have the same number of timesteps but this was not the case"
            )
        return time_series_samples

    def _convert_data_to_synctwin_format(self, data: Dataset, check_only: bool = False):
        assert data.temporal_targets is not None
        assert data.event_treatments is not None

        if data.event_treatments.n_features != 1:
            raise RuntimeError(
                f"{SyncTwinRegressor.__name__} requires exactly one event treatments feature but "
                f"{data.event_treatments.n_features} found"
            )
        # TODO: Shouldn't be limited to 1D targets, generalize.
        if data.temporal_targets.n_features != 1:
            raise RuntimeError(
                f"{SyncTwinRegressor.__name__} requires exactly one temporal targets feature but "
                f"{data.temporal_targets.n_features} found"
            )
        treatment_feature = list(data.event_treatments.features.values())[0]
        treatment_categories = tuple(treatment_feature.categories)
        if treatment_categories != self.expected_treatment_statuses:
            raise RuntimeError(
                f"{SyncTwinRegressor.__name__} requires the treatment feature to have the categories: "
                f"{self.expected_treatment_statuses} but {treatment_categories} found."
            )

        # Get n_{treated,control}_samples, sort control then treatment samples.
        treatment_feature_name = list(data.event_treatments.features.keys())[0]
        df = data.event_treatments.df
        treated_sample_indices = (df[df[treatment_feature_name] == self.treated_indicator]).index.get_level_values(0)
        control_sample_indices = (df[df[treatment_feature_name] == self.control_indicator]).index.get_level_values(0)
        n_treated_samples = len(treated_sample_indices)
        n_control_samples = len(control_sample_indices)
        if not (n_treated_samples <= n_control_samples):
            raise RuntimeError(
                f"{SyncTwinRegressor.__name__} requires the num. treated samples <= n control samples: "
                f"but these were {n_treated_samples} and {n_control_samples} respectively"
            )

        # Get the time index of each event.
        event_time_indexes: List = []
        for event in data.event_treatments:
            event_time_indexes.append(event.df.index.get_level_values(1))
        # Extract the pre-treatment part of the covariates and targets.
        pre_treat_temporal_covariates = self._extract_pre_treatment(
            data.temporal_covariates, event_time_indexes, "temporal covariates"
        )
        pre_treat_temporal_targets = self._extract_pre_treatment(
            data.temporal_targets, event_time_indexes, "temporal targets"
        )
        if not time_index_equal(pre_treat_temporal_covariates, pre_treat_temporal_targets):
            raise RuntimeError(
                f"{SyncTwinRegressor.__name__} requires pre-treatment covariates and targets have "
                "the same time indexes (for each sample) but this was not the case"
            )
        # Extract the post-treatment targets.
        post_treat_temporal_targets = self._extract_post_treatment(
            data.temporal_targets, event_time_indexes, "temporal targets"
        )

        if check_only is True:
            return True

        x_full_cov = pre_treat_temporal_covariates.to_torch_tensor(dtype=self.dtype, device=self.device)
        x_full_targ = pre_treat_temporal_targets.to_torch_tensor(dtype=self.dtype, device=self.device)
        x_full = torch.cat([x_full_cov, x_full_targ], dim=-1).permute(1, 0, 2)
        y_full = post_treat_temporal_targets.to_torch_tensor(dtype=self.dtype, device=self.device).permute(1, 0, 2)

        # Get time deltas.
        time_index = pre_treat_temporal_targets.to_torch_tensor_time_index(dtype=self.dtype, device=self.device)
        time_deltas = compute_deltas(time_index)
        time_deltas[:, 0, :] = 1.0
        assert time_deltas.shape[-1] == 1
        t_full = torch.cat([time_deltas] * x_full.shape[-1], dim=-1).permute(1, 0, 2)

        # Get the mask.
        mask_full = torch.ones_like(x_full)

        # Treatment value filtering.
        tr = torch.tensor(data.event_treatments.df.to_numpy()[:, 0], dtype=self.dtype, device=self.device)
        y_mask_full = torch.zeros_like(tr)
        y_mask_full[tr == self.treated_indicator] = 0.0
        y_mask_full[tr == self.control_indicator] = 1.0

        y_control = y_full[:, y_mask_full == 1.0, :]

        # Batch ind full.
        batch_ind_full = torch.tensor(data.sample_indices, dtype=torch.long, device=self.device)

        return (
            SyncTwinTensors(x_full, t_full, mask_full, batch_ind_full, y_full, y_control, y_mask_full),
            (n_treated_samples, n_control_samples),
        )

    def _split_data(
        self, synctwin_tensors: SyncTwinTensors, test_size: float
    ) -> Tuple[SyncTwinTensors, SyncTwinTensors]:
        # TODO: Make this cleaner.

        x_full = synctwin_tensors.x_full.permute(1, 0, 2).numpy()
        t_full = synctwin_tensors.t_full.permute(1, 0, 2).numpy()
        mask_full = synctwin_tensors.mask_full.permute(1, 0, 2).numpy()
        # batch_ind_full is regenerated as range.
        y_full = synctwin_tensors.y_full.permute(1, 0, 2).numpy()
        # y_control is re-done from y_full.
        y_mask_full = synctwin_tensors.y_mask_full.numpy()

        (
            x_full_train,
            x_full_val,
            t_full_train,
            t_full_val,
            mask_full_train,
            mask_full_val,
            y_full_train,
            y_full_val,
            y_mask_full_train,
            y_mask_full_val,
        ) = train_test_split(x_full, t_full, mask_full, y_full, y_mask_full, test_size=test_size, stratify=y_mask_full)

        x_full_train = torch.tensor(x_full_train, dtype=self.dtype, device=self.device).permute(1, 0, 2)
        t_full_train = torch.tensor(t_full_train, dtype=self.dtype, device=self.device).permute(1, 0, 2)
        mask_full_train = torch.tensor(mask_full_train, dtype=self.dtype, device=self.device).permute(1, 0, 2)
        y_full_train = torch.tensor(y_full_train, dtype=self.dtype, device=self.device).permute(1, 0, 2)
        y_mask_full_train = torch.tensor(y_mask_full_train, dtype=self.dtype, device=self.device)

        x_full_val = torch.tensor(x_full_val, dtype=self.dtype, device=self.device).permute(1, 0, 2)
        t_full_val = torch.tensor(t_full_val, dtype=self.dtype, device=self.device).permute(1, 0, 2)
        mask_full_val = torch.tensor(mask_full_val, dtype=self.dtype, device=self.device).permute(1, 0, 2)
        y_full_val = torch.tensor(y_full_val, dtype=self.dtype, device=self.device).permute(1, 0, 2)
        y_mask_full_val = torch.tensor(y_mask_full_val, dtype=self.dtype, device=self.device)

        y_control_train = y_full_train[:, y_mask_full_train == 1.0, :]
        y_control_val = y_full_val[:, y_mask_full_val == 1.0, :]

        # TODO: This is dodgy. Needs to be investigated (batch_ind_full regenerated as range).
        batch_ind_full_train = torch.arange(y_mask_full_train.shape[0], dtype=torch.long)
        batch_ind_full_val = torch.arange(y_mask_full_val.shape[0], dtype=torch.long)

        return (
            SyncTwinTensors(
                x_full=x_full_train,
                t_full=t_full_train,
                mask_full=mask_full_train,
                batch_ind_full=batch_ind_full_train,
                y_full=y_full_train,
                y_control=y_control_train,
                y_mask_full=y_mask_full_train,
            ),
            SyncTwinTensors(
                x_full=x_full_val,
                t_full=t_full_val,
                mask_full=mask_full_val,
                batch_ind_full=batch_ind_full_val,
                y_full=y_full_val,
                y_control=y_control_val,
                y_mask_full=y_mask_full_val,
            ),
        )

    def _init_submodules(self) -> None:
        self.encoder = RegularEncoder(
            input_dim=self.inferred_params.encoder_input_size, hidden_dim=self.params.hidden_size, device=self.device
        )
        self.decoder = RegularDecoder(
            hidden_dim=self.encoder.hidden_dim,
            output_dim=self.encoder.input_dim,
            max_seq_len=self.inferred_params.pre_treat_len,
            device=self.device,
        )
        self.decoder_y = LinearDecoder(
            hidden_dim=self.encoder.hidden_dim,
            output_dim=self.inferred_params.decoder_y_output_size,
            max_seq_len=self.inferred_params.post_treat_len,
            device=self.device,
        )
        self.synctwin = SyncTwinModule(
            n_unit=self.inferred_params.synctwin_n_unit,
            n_treated=self.inferred_params.synctwin_n_treated,
            device=self.device,
            dtype=self.dtype,
            reg_B=self._reg_B,
            lam_express=self._lambda_express,
            lam_recon=self.params.lambda_reconstruction,
            lam_prognostic=self.params.lambda_prognostic,
            tau=self.params.tau,
            encoder=self.encoder,
            decoder=self.decoder,
            decoder_Y=self.decoder_y,
        )

    def _init_inferred_params(self, data: Dataset, **kwargs) -> None:
        # All have already been set in _prep_data_for_fit().
        pass

    def _init_optimizers(self) -> None:
        # Handled elsewhere.
        pass

    def _prep_data_for_fit(self, data: Dataset, **kwargs):
        synctwin_tensors, _ = self._convert_data_to_synctwin_format(data)  # type: ignore
        if self.params.use_validation_set_in_training:
            synctwin_tensors_tain, synctwin_tensors_val = self._split_data(synctwin_tensors, self._validation_set_frac)
        else:
            synctwin_tensors_tain = synctwin_tensors
            synctwin_tensors_val = synctwin_tensors

        if synctwin_tensors_tain.x_full.shape[1] != synctwin_tensors_val.x_full.shape[1]:
            raise RuntimeError("Was not possible to split data into test and validation set 50:50")

        self.inferred_params.encoder_input_size = synctwin_tensors_tain.x_full.shape[-1]
        self.inferred_params.pre_treat_len = synctwin_tensors_tain.x_full.shape[0]
        self.inferred_params.decoder_y_output_size = synctwin_tensors_tain.y_full.shape[-1]
        self.inferred_params.post_treat_len = synctwin_tensors_tain.y_full.shape[0]
        self.inferred_params.synctwin_n_unit = (synctwin_tensors_tain.y_mask_full == 1.0).sum().item()
        self.inferred_params.synctwin_n_treated = (synctwin_tensors_tain.y_mask_full == 0.0).sum().item()

        return synctwin_tensors_tain, synctwin_tensors_val

    def _prep_submodules_for_fit(self) -> None:
        if TYPE_CHECKING:
            assert self.encoder is not None and self.decoder is not None
            assert self.decoder_y is not None
            assert self.synctwin is not None
        self.encoder.to(self.device, dtype=self.dtype)
        self.decoder.to(self.device, dtype=self.dtype)
        self.decoder_y.to(self.device, dtype=self.dtype)
        self.synctwin.to(self.device, dtype=self.dtype)
        self.encoder.train()
        self.decoder.train()
        self.decoder_y.train()
        self.synctwin.train()

    def _prep_data_for_predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _prep_submodules_for_predict(self) -> None:
        raise NotImplementedError

    def _prep_data_for_predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Optional[Horizon],
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        assert data.event_treatments is not None
        assert len(treatment_scenarios) == 1
        assert treatment_scenarios[0].df.shape == (1, 1)
        treatment_indicator_scenario = treatment_scenarios[0].df.values[0, 0]
        treatment_indicator_actual = data.event_treatments[sample_index].df.values[0, 0]
        if treatment_indicator_scenario == treatment_indicator_actual:
            raise ValueError(
                f"Factual treatment indicator ({treatment_indicator_scenario}) for "
                "sample {sample_index} cannot be provided as a treatment scenario"
            )
        if treatment_indicator_actual == self.control_indicator:
            raise ValueError(
                "Currently can only predict counterfactuals for *treated* samples "
                "(i.e. can predict the untreated outcome for a sample that factually received treatment). "
                f"However sample {sample_index} was a control sample in the data.\n{treatment_scenarios[0]}"
            )

        synctwin_tensors, _ = self._convert_data_to_synctwin_format(data)  # type: ignore

        assert self.inferred_params.encoder_input_size == synctwin_tensors.x_full.shape[-1]
        assert self.inferred_params.pre_treat_len == synctwin_tensors.x_full.shape[0]
        assert self.inferred_params.decoder_y_output_size == synctwin_tensors.y_full.shape[-1]
        assert self.inferred_params.post_treat_len == synctwin_tensors.y_full.shape[0]
        self._predict_synctwin_n_unit = (synctwin_tensors.y_mask_full == 1.0).sum().item()
        self._predict_synctwin_n_treated = (synctwin_tensors.y_mask_full == 0.0).sum().item()

        return synctwin_tensors

    def _prep_submodules_for_predict_counterfactuals(self) -> None:
        # NOTE: This whole method content - not intuitive.
        assert self.encoder is not None and self.decoder is not None and self.decoder_y is not None
        assert self._predict_synctwin_n_unit is not None and self._predict_synctwin_n_treated is not None
        self.synctwin = SyncTwinModule(
            n_unit=self._predict_synctwin_n_unit,
            n_treated=self._predict_synctwin_n_treated,
            device=self.device,
            dtype=self.dtype,
            reg_B=self._reg_B,
            lam_express=self._lambda_express,
            lam_recon=self.params.lambda_reconstruction,
            lam_prognostic=self.params.lambda_prognostic,
            tau=self.params.tau,
            encoder=self.encoder,
            decoder=self.decoder,
            decoder_Y=self.decoder_y,
        )
        self.encoder.to(self.device, dtype=self.dtype)
        self.decoder.to(self.device, dtype=self.dtype)
        self.decoder_y.to(self.device, dtype=self.dtype)
        self.synctwin.to(self.device, dtype=self.dtype)
        self.encoder.train()
        self.decoder.train()
        self.decoder_y.train()
        self.synctwin.train()

    def _fit(self, data: Dataset, horizon: Optional[Horizon] = None, **kwargs) -> "SyncTwinRegressor":
        self.set_attributes_from_kwargs(**kwargs)

        synctwin_tensors_tain, synctwin_tensors_val = self.prep_fit(data)
        assert isinstance(synctwin_tensors_tain, SyncTwinTensors)
        assert isinstance(synctwin_tensors_val, SyncTwinTensors)

        print("=== Training Stage 1: Pretraining ===")
        synctwin_train_utils.pre_train_reconstruction_prognostic_loss(
            self.synctwin,
            x_full=synctwin_tensors_tain.x_full,
            t_full=synctwin_tensors_tain.t_full,
            mask_full=synctwin_tensors_tain.mask_full,
            y_full=synctwin_tensors_tain.y_full,
            y_mask_full=synctwin_tensors_tain.y_mask_full,
            x_full_val=synctwin_tensors_val.x_full,
            t_full_val=synctwin_tensors_val.t_full,
            mask_full_val=synctwin_tensors_val.mask_full,
            y_full_val=synctwin_tensors_val.y_full,
            y_mask_full_val=synctwin_tensors_val.y_mask_full,
            niters=self.params.pretraining_iterations,
            batch_size=self.params.batch_size,
            test_freq=self._pretraining_test_freq,
        )

        print("=== Training Stage 2: Matching ===")
        synctwin_train_utils.update_representations(
            self.synctwin,
            x_full=synctwin_tensors_val.x_full,
            t_full=synctwin_tensors_val.t_full,
            mask_full=synctwin_tensors_val.mask_full,
            batch_ind_full=synctwin_tensors_val.batch_ind_full,
        )
        synctwin_train_utils.train_B_self_expressive(
            self.synctwin,
            x_full=synctwin_tensors_val.x_full,
            t_full=synctwin_tensors_val.t_full,
            mask_full=synctwin_tensors_val.mask_full,
            batch_ind_full=synctwin_tensors_val.batch_ind_full,
            niters=self.params.matching_iterations,
            batch_size=None,  # NOTE: Batched training not implemented.
            lr=self._self_expressive_lr,
            test_freq=self._matching_test_freq,
        )

        return self

    def _predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> TPredictOutput:
        raise NotImplementedError(
            "predict() method of SyncTwin is not implemented. To get counterfactual "
            "predictions, call predict_counterfactuals()"
        )

    def get_possible_prediction_horizon(self, sample_index: T_SamplesIndexDtype, data: Dataset):
        self._convert_data_to_synctwin_format(data, check_only=True)
        # If the above validates fine, get the horizon:
        assert data.temporal_targets is not None
        assert data.event_treatments is not None
        sample_targets_timeseries = data.temporal_targets[sample_index]
        sample_event_time = data.event_treatments.df.loc[[sample_index], :, :].index.get_level_values(1)[0]  # type: ignore
        future_horizon = sample_targets_timeseries.df.loc[sample_event_time:, :].index
        return TimeIndexHorizon(time_index_sequence=[future_horizon])

    def get_possible_treatment_scenarios(self, sample_index: T_SamplesIndexDtype, data: Dataset):
        self._convert_data_to_synctwin_format(data, check_only=True)
        assert data.event_treatments is not None
        sample_event = data.event_treatments[sample_index]
        # print(sample_event)
        sample_event_treatment_indicator = sample_event.df.values[0, 0]
        new_indicator = self._get_other_treatment_status(sample_event_treatment_indicator)
        sample_event = sample_event.copy()
        sample_event.df[:] = new_indicator
        return (sample_event,)

    def _predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Optional[Horizon],
        **kwargs,
    ) -> TCounterfactualPredictions:
        self.set_attributes_from_kwargs(**kwargs)

        assert data.temporal_targets is not None
        assert isinstance(horizon, TimeIndexHorizon)
        assert self.synctwin is not None

        synctwin_tensors = self.prep_predict_counterfactuals(data, sample_index, treatment_scenarios, horizon)
        assert isinstance(synctwin_tensors, SyncTwinTensors)

        print("=== Running Inference Stage 1: Matching ===")
        synctwin_train_utils.update_representations(
            self.synctwin,
            x_full=synctwin_tensors.x_full,
            t_full=synctwin_tensors.t_full,
            mask_full=synctwin_tensors.mask_full,
            batch_ind_full=synctwin_tensors.batch_ind_full,
        )
        synctwin_train_utils.train_B_self_expressive(
            self.synctwin,
            x_full=synctwin_tensors.x_full,
            t_full=synctwin_tensors.t_full,
            mask_full=synctwin_tensors.mask_full,
            batch_ind_full=synctwin_tensors.batch_ind_full,
            niters=self.params.inference_iterations,
            batch_size=None,  # NOTE: Batched training not implemented.
            lr=self._self_expressive_lr,
            test_freq=self._inference_iterations_test_freq,
        )
        synctwin_train_utils.update_representations(
            self.synctwin,
            x_full=synctwin_tensors.x_full,
            t_full=synctwin_tensors.t_full,
            mask_full=synctwin_tensors.mask_full,
            batch_ind_full=synctwin_tensors.batch_ind_full,
        )
        self.synctwin.eval()

        print("=== Running Inference Stage 2: Computing Counterfactuals ===")
        y_hat = synctwin_train_utils.get_prediction(
            self.synctwin,
            batch_ind_full=synctwin_tensors.batch_ind_full,
            y_control=synctwin_tensors.y_control,
            itr=self._prediction_compute_iters,
        )
        print("Done")
        y_hat_sample = y_hat[:, sample_index, :]

        y_hat_sample = y_hat_sample.detach().cpu().numpy()
        data_historic_temporal_targets = data.temporal_targets[sample_index]
        if TYPE_CHECKING:
            assert isinstance(data_historic_temporal_targets, TimeSeries)
        list_ts = to_counterfactual_predictions([y_hat_sample], data_historic_temporal_targets, horizon)

        return list_ts
