from typing import TYPE_CHECKING, Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..components.torch.common import OPTIM_MAP
from ..components.torch.ffnn import FeedForwardNet
from ..components.torch.gradient_reversal import GradientReversalModule
from ..components.torch.interfaces import OrganizedTreatmentEffectsModuleMixin
from ..components.torch.rnn import RecurrentFFNet, apply_to_each_timestep, mask_and_reshape
from ..data import DEFAULT_PADDING_INDICATOR, Dataset, TimeSeries, TimeSeriesSamples
from ..data.constants import T_SamplesIndexDtype
from ..data.utils import split_time_series, time_index_utils, to_counterfactual_predictions
from ..interface import (
    Horizon,
    TCounterfactualPredictions,
    TDefaultParams,
    TimeIndexHorizon,
    TParams,
    TreatmentEffectsModel,
    TTreatmentScenarios,
)
from ..interface import requirements as r
from ..prediction.seq2seq import Seq2SeqCRNStylePredictorBase
from ..utils import tensor_like as tl
from ..utils.array_manipulation import n_step_shift_back, n_step_shift_forward
from ..utils.dev import NEEDED

_DEBUG = False


# TODO: For clarity, get rid of TEncodedRepresentation and always use RNNHidden?
TEncodedRepresentation = Tuple[torch.Tensor, Optional[torch.Tensor]]


class RecurrentFFNet_ConcatTreatment(RecurrentFFNet):
    def rnn_out_postprocess(self, rnn_out: torch.Tensor, **kwargs) -> torch.Tensor:
        concat_treatment = kwargs["concat_treatment"]
        return torch.cat([rnn_out, concat_treatment], dim=-1)

    def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
        concat_treatment = kwargs.pop("concat_treatment")
        concat_treatment = concat_treatment[:, [timestep_idx], :]
        out, *_ = self.forward(x, concat_treatment=concat_treatment, **kwargs)
        return out


class TreatBalancerNet(FeedForwardNet):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = tuple(),
        out_activation: Optional[str] = "ReLU",
        hidden_activations: Optional[str] = "ReLU",
    ) -> None:
        super().__init__(in_dim, out_dim, hidden_dims, out_activation, hidden_activations)
        self.softmax = nn.Softmax(dim=-1)
        self.revgrad = GradientReversalModule(alpha=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revgrad(x)
        x = super().forward(x)
        x = self.softmax(x)
        return x


class _DefaultParams(NamedTuple):
    # Encoder:
    encoder_rnn_type: str = "LSTM"
    encoder_hidden_size: int = 100
    encoder_num_layers: int = 1
    encoder_bias: bool = True
    encoder_dropout: float = 0.0
    encoder_bidirectional: bool = False
    encoder_nonlinearity: Optional[str] = None
    encoder_proj_size: Optional[int] = None
    # Decoder:
    decoder_rnn_type: str = "LSTM"
    decoder_hidden_size: int = 100
    decoder_num_layers: int = 1
    decoder_bias: bool = True
    decoder_dropout: float = 0.0
    decoder_bidirectional: bool = False
    decoder_nonlinearity: Optional[str] = None
    decoder_proj_size: Optional[int] = None
    # Adapter FF NN:
    adapter_hidden_dims: Sequence[int] = [50]
    adapter_out_activation: Optional[str] = "Tanh"
    # Predictor FF NN:
    predictor_hidden_dims: Sequence[int] = []
    predictor_out_activation: Optional[str] = None
    # Treatment Balancer FF NN:
    treat_net_hidden_dims: Sequence[int] = []
    treat_net_out_activation: Optional[str] = None
    # Misc:
    max_len: Optional[int] = None
    optimizer_str: str = "Adam"
    optimizer_kwargs: Mapping[str, Any] = dict(lr=0.01, weight_decay=1e-5)
    batch_size: int = 32
    epochs: int = 100
    padding_indicator: float = DEFAULT_PADDING_INDICATOR


# TODO: Test this with various sets of params and make sure it doesn't fail.
class CRNTreatmentEffectsModelBase(
    TreatmentEffectsModel, Seq2SeqCRNStylePredictorBase, OrganizedTreatmentEffectsModuleMixin
):
    requirements: r.Requirements
    DEFAULT_PARAMS: TDefaultParams

    def __init__(self, loss_fn: nn.Module, params: Optional[TParams] = None) -> None:
        TreatmentEffectsModel.__init__(self, params)
        Seq2SeqCRNStylePredictorBase.__init__(self, loss_fn=loss_fn, params=params)

        # Treatment balancer.
        self.encoder_treat_net: Optional[FeedForwardNet] = NEEDED
        self.decoder_treat_net: Optional[FeedForwardNet] = NEEDED

        self.treat_loss = nn.CrossEntropyLoss()

    def _init_submodules_encoder_decoder(self) -> None:
        # Initialize Encoder models:
        self.encoder = RecurrentFFNet_ConcatTreatment(
            rnn_type=self.params.encoder_rnn_type,
            input_size=self.inferred_params.encoder_input_size,
            hidden_size=self.params.encoder_hidden_size,
            nonlinearity=self.params.encoder_nonlinearity,
            num_layers=self.params.encoder_num_layers,
            bias=self.params.encoder_bias,
            dropout=self.params.encoder_dropout,
            bidirectional=self.params.encoder_bidirectional,
            proj_size=self.params.encoder_proj_size,
            ff_out_size=self.inferred_params.encoder_predictor_output_size,
            ff_in_size_adjust=self.inferred_params.predictor_input_size_adjust,  # Note this.
            ff_hidden_dims=self.params.predictor_hidden_dims,
            ff_out_activation=self.params.predictor_out_activation,
            ff_hidden_activations="ReLU",
        )

        # Initialize Decoder models:
        self.decoder = RecurrentFFNet_ConcatTreatment(
            rnn_type=self.params.decoder_rnn_type,
            input_size=self.inferred_params.decoder_input_size,
            hidden_size=self.params.decoder_hidden_size,
            nonlinearity=self.params.decoder_nonlinearity,
            num_layers=self.params.decoder_num_layers,
            bias=self.params.decoder_bias,
            dropout=self.params.decoder_dropout,
            bidirectional=self.params.decoder_bidirectional,
            proj_size=self.params.decoder_proj_size,
            ff_out_size=self.inferred_params.decoder_predictor_output_size,
            ff_in_size_adjust=self.inferred_params.predictor_input_size_adjust,  # Note this.
            ff_hidden_dims=self.params.predictor_hidden_dims,
            ff_out_activation=self.params.predictor_out_activation,
            ff_hidden_activations="ReLU",
        )

    def _init_submodules_treat_net(self) -> None:
        # Initialize Treatment Balancers:
        if TYPE_CHECKING:
            assert self.encoder is not None and self.decoder is not None

        encoder_out_dim, *_ = self.encoder.rnn.get_output_and_h_dim()
        decoder_out_dim, *_ = self.decoder.rnn.get_output_and_h_dim()
        self.inferred_params.encoder_treat_net_input_size = encoder_out_dim
        self.inferred_params.decoder_treat_net_input_size = decoder_out_dim

        self.encoder_treat_net = TreatBalancerNet(
            in_dim=self.inferred_params.encoder_treat_net_input_size,
            out_dim=self.inferred_params.treat_net_output_size,
            hidden_dims=self.params.treat_net_hidden_dims,
            out_activation=self.params.treat_net_out_activation,
            hidden_activations="ReLU",
        )
        self.decoder_treat_net = TreatBalancerNet(
            in_dim=self.inferred_params.decoder_treat_net_input_size,
            out_dim=self.inferred_params.treat_net_output_size,
            hidden_dims=self.params.treat_net_hidden_dims,
            out_activation=self.params.treat_net_out_activation,
            hidden_activations="ReLU",
        )

    def _init_submodules(self) -> None:
        self._init_submodules_encoder_decoder()
        super()._init_submodules_adapter()
        self._init_submodules_treat_net()

    def _init_inferred_params(self, data: Dataset, **kwargs) -> None:
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None
        assert data.temporal_treatments is not None

        # Initialize the helper attributes.
        # + 1 below are for time deltas.
        self.inferred_params.encoder_input_size = (
            data.temporal_covariates.n_features + data.temporal_treatments.n_features + 1
        )
        if data.static_covariates is not None:
            self.inferred_params.encoder_input_size += data.static_covariates.n_features
        self.inferred_params.decoder_input_size = (
            data.temporal_targets.n_features + data.temporal_treatments.n_features + 1
        )
        if data.static_covariates is not None:
            self.inferred_params.decoder_input_size += data.static_covariates.n_features
        self.inferred_params.encoder_input_size += data.temporal_targets.n_features
        self.inferred_params.encoder_predictor_output_size = data.temporal_targets.n_features
        self.inferred_params.decoder_predictor_output_size = data.temporal_targets.n_features
        self.inferred_params.treat_net_output_size = data.temporal_treatments.n_features
        self.inferred_params.predictor_input_size_adjust = data.temporal_treatments.n_features

        # Inferred batch size:
        self.inferred_params.encoder_batch_size = min(self.params.batch_size, data.n_samples)
        self.inferred_params.decoder_batch_size = NEEDED  # This is set later.

    def _init_optimizers(self):
        self.encoder = cast(RecurrentFFNet, self.encoder)
        self.encoder_treat_net = cast(FeedForwardNet, self.encoder_treat_net)
        self.adapter = cast(FeedForwardNet, self.adapter)
        self.decoder = cast(RecurrentFFNet, self.decoder)
        self.decoder_treat_net = cast(FeedForwardNet, self.decoder_treat_net)
        # Initialize optimizers.
        self.encoder_optim = OPTIM_MAP[self.params.optimizer_str](
            params=[*self.encoder.parameters(), *self.encoder_treat_net.parameters()],
            **self.params.optimizer_kwargs,
        )
        self.decoder_optim = OPTIM_MAP[self.params.optimizer_str](
            params=[*self.adapter.parameters(), *self.decoder.parameters(), *self.decoder_treat_net.parameters()],
            **self.params.optimizer_kwargs,
        )

    def _prep_treat_tensors(self, data: Dataset, t_cov: torch.Tensor, t_targ: torch.Tensor):
        if TYPE_CHECKING:
            assert data.temporal_treatments is not None

        t_treat = data.temporal_treatments.to_torch_tensor(
            padding_indicator=self.params.padding_indicator,
            max_len=self.params.max_len,
            dtype=self.dtype,
            device=self.device,
        )
        t_cov = n_step_shift_back(t_cov, n_step=1)
        t_targ = n_step_shift_back(t_targ, n_step=1)
        t_treat_out = n_step_shift_back(t_treat, n_step=1)[:, : t_cov.shape[1], :]
        t_treat = n_step_shift_forward(t_treat, n_step=1)[:, : t_cov.shape[1], :]
        t_cov = torch.cat([t_cov, t_treat], dim=-1)

        # NOTE:
        # If time indexes originally are:
        # t_targ like        [1, 2, 3, 4]
        # t_cov like         [0, 1, 2, 3]
        # t_treat like       [0, 1, 2, 3, (...)]
        #
        # Then at the end:
        # t_targ like        [2, 3, 4]
        # t_treat_out like   [1, 2, 3]
        # New t_cov combines:
        #   original t_cov   [1, 2, 3]
        #   original t_treat [0, 1, 2]

        return t_cov, t_targ, t_treat_out

    def _prep_torch_tensors_encoder(self, data: Dataset, shift_targ_cov: bool):
        t_cov, t_targ = super()._prep_torch_tensors_encoder(data, shift_targ_cov)
        t_cov, t_targ, t_treat_out = self._prep_treat_tensors(data, t_cov=t_cov, t_targ=t_targ)
        return t_cov, t_targ, t_treat_out

    def _prep_torch_tensors_decoder(self, data: Dataset):
        t_targ, decoder_input = super()._prep_torch_tensors_decoder(data)
        decoder_input, t_targ, t_treat_out = self._prep_treat_tensors(data, t_cov=decoder_input, t_targ=t_targ)
        return t_targ, decoder_input, t_treat_out

    def _prep_data_for_fit(self, data: Dataset, **kwargs) -> Tuple[torch.Tensor, ...]:
        min_pre_len = kwargs.pop("min_pre_len")
        min_post_len = kwargs.pop("min_post_len")
        repeat_last_pre_step = kwargs.pop("repeat_last_pre_step")

        encoder_tensors = self._prep_torch_tensors_encoder(data, shift_targ_cov=True)

        print("Preparing data for decoder training...")
        data_pre, data_post, _ = split_time_series.split_at_each_step(
            data, min_pre_len=min_pre_len, min_post_len=min_post_len, repeat_last_pre_step=repeat_last_pre_step
        )
        self.inferred_params.decoder_batch_size = min(self.params.batch_size, data_post.n_samples)
        print("Preparing data for decoder training DONE.")

        t_cov_to_encode, _, t_treat_to_encode = self._prep_torch_tensors_encoder(data_pre, shift_targ_cov=False)

        decoder_tensors = self._prep_torch_tensors_decoder(data_post)

        return (*encoder_tensors, t_cov_to_encode, t_treat_to_encode, *decoder_tensors)

    def _prep_torch_tensors_decoder_inference(
        self,
        data: Dataset,
        horizon: TimeIndexHorizon,
    ):
        decoder_input = super()._prep_torch_tensors_decoder_inference(data, horizon)

        if TYPE_CHECKING:
            assert data.temporal_treatments is not None

        ts_treat = time_index_utils.time_series_samples.take_all_from_one_before_start(
            time_series_samples_=data.temporal_treatments, time_indexes=horizon, inplace=False
        )
        if TYPE_CHECKING:
            assert ts_treat is not None
        t_treat = ts_treat.to_torch_tensor(
            padding_indicator=self.params.padding_indicator,
            max_len=decoder_input.shape[1] + 1,
            dtype=self.dtype,
            device=self.device,
        )
        decoder_input = torch.cat([decoder_input, t_treat[:, :-1, :]], dim=-1)
        t_treat_out = t_treat[:, 1:, :]

        return decoder_input, t_treat_out

    def _prep_torch_tensors_decoder_inference_counterfactuals(
        self,
        data: Dataset,
        treatment_scenario: TimeSeries,
        horizon: TimeIndexHorizon,
    ):
        decoder_input = super()._prep_torch_tensors_decoder_inference(data, horizon)

        if TYPE_CHECKING:
            assert data.temporal_treatments is not None

        t_treat_last = time_index_utils.time_series_samples.take_one_before_start(data.temporal_treatments, horizon)
        if TYPE_CHECKING:
            assert t_treat_last is not None
        t_treat_last = t_treat_last.to_torch_tensor(
            padding_indicator=self.params.padding_indicator,
            max_len=1,
            dtype=self.dtype,
            device=self.device,
        )
        t_treat = treatment_scenario.to_torch_tensor(
            padding_indicator=self.params.padding_indicator,
            max_len=decoder_input.shape[1],
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(dim=0)
        t_treat = torch.cat([t_treat_last, t_treat], dim=1)

        decoder_input = torch.cat([decoder_input, t_treat[:, :-1, :]], dim=-1)
        t_treat_out = t_treat[:, 1:, :]

        return decoder_input, t_treat_out

    def _prep_data_for_predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> Tuple[torch.Tensor, ...]:
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None
        assert isinstance(horizon, TimeIndexHorizon)

        # Make sure to not use "future" values for prediction.
        data_encode = time_index_utils.dataset.take_temporal_data_before_start(data, horizon, inplace=False)
        if TYPE_CHECKING:
            assert data_encode is not None

        t_cov_to_encode, _, t_treat_out = self._prep_torch_tensors_encoder(data_encode, shift_targ_cov=False)
        encoded_representations = self._get_encoder_representation(t_cov_to_encode, t_treat_out=t_treat_out)
        h, c = super()._reshape_h_sample_dim_0(encoded_representations)

        decoder_input, decoder_t_treat_out = self._prep_torch_tensors_decoder_inference(data, horizon)
        assert h.shape[0] == decoder_input.shape[0]

        return h, c, decoder_input, decoder_t_treat_out

    def _prep_data_for_predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Optional[Horizon],
        **kwargs,
    ) -> Tuple[Any, ...]:
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None
        assert isinstance(horizon, TimeIndexHorizon)

        # Make sure to not use "future" values for prediction.
        data_encode = time_index_utils.dataset.take_temporal_data_before_start(data, horizon, inplace=False)
        if TYPE_CHECKING:
            assert data_encode is not None

        t_cov_to_encode, _, t_treat_out = self._prep_torch_tensors_encoder(data_encode, shift_targ_cov=False)
        encoded_representations = self._get_encoder_representation(t_cov_to_encode, t_treat_out=t_treat_out)
        h, c = super()._reshape_h_sample_dim_0(encoded_representations)

        decoder_input_list = []
        decoder_t_treat_out_list = []
        for treatment_scenario in treatment_scenarios:
            assert isinstance(treatment_scenario, TimeSeries)
            decoder_input, decoder_t_treat_out = self._prep_torch_tensors_decoder_inference_counterfactuals(
                data, treatment_scenario, horizon
            )
            assert h.shape[0] == decoder_input.shape[0]
            decoder_input_list.append(decoder_input)
            decoder_t_treat_out_list.append(decoder_t_treat_out)

        return h, c, decoder_input_list, decoder_t_treat_out_list

    def _prep_submodules_for_fit(self) -> None:
        assert self.encoder_treat_net is not None and self.decoder_treat_net is not None
        super()._prep_submodules_for_fit()
        self.encoder_treat_net.to(self.device, dtype=self.dtype)
        self.decoder_treat_net.to(self.device, dtype=self.dtype)
        self.encoder_treat_net.train()
        self.decoder_treat_net.train()

    def _prep_submodules_for_predict(self) -> None:
        assert self.encoder_treat_net is not None and self.decoder_treat_net is not None
        super()._prep_submodules_for_predict()
        self.encoder_treat_net.to(self.device, dtype=self.dtype)
        self.decoder_treat_net.to(self.device, dtype=self.dtype)
        self.encoder_treat_net.eval()
        self.decoder_treat_net.eval()

    def _compute_lambda(self, epoch_idx: int) -> torch.Tensor:
        return 2.0 / (1.0 + torch.exp(-10.0 * torch.tensor(epoch_idx + 1))) - 1.0

    def _train_encoder(self, encoder_tensors: Tuple) -> None:
        if TYPE_CHECKING:
            assert self.encoder is not None
            assert self.encoder_optim is not None
            assert self.encoder_treat_net is not None

        dataloader = DataLoader(
            TensorDataset(*encoder_tensors), batch_size=self.inferred_params.encoder_batch_size, shuffle=True
        )

        for epoch_idx in range(self.params.epochs):
            n_samples_cumul = 0
            epoch_loss = 0.0
            epoch_loss_target = 0.0
            epoch_loss_treat = 0.0
            lambda_ = 0.0
            for _, (t_cov, t_targ, t_treat_out) in enumerate(dataloader):
                current_batch_size = t_cov.shape[0]
                n_samples_cumul += current_batch_size

                out, rnn_out, _ = self.encoder(
                    t_cov, h=None, padding_indicator=self.params.padding_indicator, concat_treatment=t_treat_out
                )
                out_treat_net = apply_to_each_timestep(
                    self.encoder_treat_net,
                    input_tensor=rnn_out,
                    output_size=self.inferred_params.treat_net_output_size,
                    concat_tensors=[],
                    padding_indicator=self.params.padding_indicator,
                    expected_module_input_size=self.inferred_params.encoder_treat_net_input_size,
                )

                not_padding_targ = ~tl.eq_indicator(t_targ, self.params.padding_indicator)
                if TYPE_CHECKING:
                    assert isinstance(not_padding_targ, torch.BoolTensor)
                out = mask_and_reshape(mask_selector=not_padding_targ, tensor=out)
                t_targ = mask_and_reshape(mask_selector=not_padding_targ, tensor=t_targ)

                not_padding_treat = ~tl.eq_indicator(t_treat_out, self.params.padding_indicator)
                out_treat_net = mask_and_reshape(mask_selector=not_padding_treat, tensor=out_treat_net)
                t_treat_out = mask_and_reshape(mask_selector=not_padding_treat, tensor=t_treat_out)

                out = self.process_output_for_loss(out)
                loss_target = self.loss_fn(out, t_targ)
                loss_treat = self.treat_loss(out_treat_net, t_treat_out)

                lambda_ = self._compute_lambda(epoch_idx)
                loss = loss_target + lambda_ * loss_treat

                # Optimization:
                self.encoder_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()

                epoch_loss_target += loss_target.item() * current_batch_size
                epoch_loss_treat += loss_treat.item() * current_batch_size
                epoch_loss += loss.item() * current_batch_size

            epoch_loss_target /= n_samples_cumul
            epoch_loss_treat /= n_samples_cumul
            epoch_loss /= n_samples_cumul
            print(
                f"Epoch: {epoch_idx}, Prediction Loss: {epoch_loss_target:.3f}, "
                f"Lambda: {lambda_:.3f}, Treatment BR Loss: {epoch_loss_treat:.3f}, Loss: {epoch_loss:.3f}"
            )

    def _get_encoder_representation(self, t_cov: torch.Tensor, t_treat_out=NEEDED, **kwargs) -> TEncodedRepresentation:
        # 2. Get the encoded representations.
        if TYPE_CHECKING:
            assert self.encoder is not None
        assert isinstance(t_treat_out, torch.Tensor)

        # Not sure this is needed here, but just in case:
        self.encoder.eval()

        is_lstm = self.params.encoder_rnn_type == "LSTM"
        with torch.no_grad():
            _, _, h = self.encoder(
                t_cov, h=None, padding_indicator=self.params.padding_indicator, concat_treatment=t_treat_out
            )
            h, c = h if is_lstm else (h, None)

        return h, c

    def _train_decoder(self, encoded_representations: TEncodedRepresentation, decoder_tensors: Tuple) -> None:
        if TYPE_CHECKING:
            assert self.encoder is not None and self.decoder is not None
            assert self.decoder_optim is not None
            assert self.decoder_treat_net is not None
            assert self.adapter is not None

        (t_targ, decoder_input, t_treat_out) = decoder_tensors
        h, c = super()._reshape_h_sample_dim_0(encoded_representations)
        assert h.shape[0] == decoder_input.shape[0]

        dataloader = DataLoader(
            TensorDataset(h, c, t_targ, decoder_input, t_treat_out),
            batch_size=self.inferred_params.decoder_batch_size,
            shuffle=True,
        )

        for epoch_idx in range(self.params.epochs):
            n_samples_cumul = 0
            epoch_loss = 0.0
            epoch_loss_target = 0.0
            epoch_loss_treat = 0.0
            lambda_ = 0.0
            for _, (h, c, t_targ, decoder_input, t_treat_out) in enumerate(dataloader):
                current_batch_size = t_targ.shape[0]
                n_samples_cumul += current_batch_size

                # Pass encoded representations through the adapter.
                h_adapter_out = self._pass_h_through_adapter(h, c)

                out, rnn_out, _ = self.decoder(
                    decoder_input,
                    h=h_adapter_out,
                    padding_indicator=self.params.padding_indicator,
                    concat_treatment=t_treat_out,
                )
                out_treat_net = apply_to_each_timestep(
                    self.decoder_treat_net,
                    input_tensor=rnn_out,
                    output_size=self.inferred_params.treat_net_output_size,
                    concat_tensors=[],
                    padding_indicator=self.params.padding_indicator,
                    expected_module_input_size=self.inferred_params.decoder_treat_net_input_size,
                )

                not_padding_targ = ~tl.eq_indicator(t_targ, self.params.padding_indicator)
                if TYPE_CHECKING:
                    assert isinstance(not_padding_targ, torch.BoolTensor)
                out = mask_and_reshape(mask_selector=not_padding_targ, tensor=out)
                t_targ = mask_and_reshape(mask_selector=not_padding_targ, tensor=t_targ)

                not_padding_treat = ~tl.eq_indicator(t_treat_out, self.params.padding_indicator)
                out_treat_net = mask_and_reshape(mask_selector=not_padding_treat, tensor=out_treat_net)
                t_treat_out = mask_and_reshape(mask_selector=not_padding_treat, tensor=t_treat_out)

                out = self.process_output_for_loss(out)
                loss_target = self.loss_fn(out, t_targ)
                loss_treat = self.treat_loss(out_treat_net, t_treat_out)

                lambda_ = self._compute_lambda(epoch_idx)
                loss = loss_target + lambda_ * loss_treat

                # Optimization:
                self.decoder_optim.zero_grad()
                loss.backward()
                self.decoder_optim.step()

                epoch_loss_target += loss_target.item() * current_batch_size
                epoch_loss_treat += loss_treat.item() * current_batch_size
                epoch_loss += loss.item() * current_batch_size

            epoch_loss_target /= n_samples_cumul
            epoch_loss_treat /= n_samples_cumul
            epoch_loss /= n_samples_cumul
            print(
                f"Epoch: {epoch_idx}, Prediction Loss: {epoch_loss_target:.3f}, "
                f"Lambda: {lambda_:.3f}, Treatment BR Loss: {epoch_loss_treat:.3f}, Loss: {epoch_loss:.3f}"
            )

    def _fit(
        self,
        data: Dataset,
        horizon: Horizon = None,  # type: ignore
        **kwargs,
    ) -> "CRNTreatmentEffectsModelBase":
        self.set_attributes_from_kwargs(**kwargs)

        # Ensure there are at least 3 timesteps in the "post" part of TimeSeries after the split and
        # at least 3 timesteps in the "pre" part. This is due to the cov./targ./treat. shifts that are needed.
        (
            encoder_t_cov,
            encoder_t_targ,
            encoder_t_treat_out,
            t_cov_to_encode,
            t_treat_to_encode,
            decoder_t_targ,
            decoder_input,
            decoder_t_treat_out,
        ) = self.prep_fit(data=data, min_pre_len=3, min_post_len=3, repeat_last_pre_step=True)

        # Run the training stages.
        print("=== Training stage: 1. Train encoder ===")
        self._train_encoder(encoder_tensors=(encoder_t_cov, encoder_t_targ, encoder_t_treat_out))

        print("=== Training stage: 2. Train decoder ===")
        if TYPE_CHECKING:
            assert isinstance(t_cov_to_encode, torch.Tensor)
        encoded_representations = self._get_encoder_representation(t_cov_to_encode, t_treat_out=t_treat_to_encode)
        self._train_decoder(
            encoded_representations, decoder_tensors=(decoder_t_targ, decoder_input, decoder_t_treat_out)
        )

        return self

    def _predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> TimeSeriesSamples:
        self.set_attributes_from_kwargs(**kwargs)

        data = data.copy()
        if TYPE_CHECKING:
            assert self.decoder is not None
            assert data.temporal_targets is not None
            assert isinstance(horizon, TimeIndexHorizon)

        h, c, decoder_input, t_treat_out = self.prep_predict(data, horizon=horizon)
        if TYPE_CHECKING:
            assert isinstance(h, torch.Tensor)
            assert isinstance(decoder_input, torch.Tensor)
            assert isinstance(t_treat_out, torch.Tensor)

        with torch.no_grad():
            h_adapter_out = self._pass_h_through_adapter(h, c)

            out = self.decoder.autoregress(
                decoder_input,
                h=h_adapter_out,
                padding_indicator=self.params.padding_indicator,
                concat_treatment=t_treat_out,
            )

            out_final: Any = self.process_output_for_loss(out)
            out_final[tl.eq_indicator(out, self.params.padding_indicator)] = self.params.padding_indicator

        prediction = TimeSeriesSamples.new_empty_like(like=data.temporal_targets)
        prediction.update_from_sequence_of_arrays(
            out_final, time_index_sequence=horizon.time_index_sequence, padding_indicator=self.params.padding_indicator
        )
        return prediction

    def _predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Optional[Horizon],
        **kwargs,
    ) -> TCounterfactualPredictions:
        self.set_attributes_from_kwargs(**kwargs)

        data = data[sample_index].copy()
        if TYPE_CHECKING:
            assert self.decoder is not None
            assert data.temporal_targets is not None
            assert isinstance(horizon, TimeIndexHorizon)

        h, c, decoder_input_list, decoder_t_treat_out_list = self.prep_predict_counterfactuals(
            data, sample_index=sample_index, treatment_scenarios=treatment_scenarios, horizon=horizon
        )
        if TYPE_CHECKING:
            assert isinstance(h, torch.Tensor)
            assert isinstance(decoder_input_list, list)
            assert isinstance(decoder_t_treat_out_list, list)

        list_counterfactual_predictions: List[torch.Tensor] = []
        for decoder_input, decoder_t_treat_out in zip(decoder_input_list, decoder_t_treat_out_list):
            assert isinstance(decoder_input, torch.Tensor)
            assert isinstance(decoder_t_treat_out, torch.Tensor)

            with torch.no_grad():
                h_adapter_out = self._pass_h_through_adapter(h, c)

                out = self.decoder.autoregress(
                    decoder_input,
                    h=h_adapter_out,
                    padding_indicator=self.params.padding_indicator,
                    concat_treatment=decoder_t_treat_out,
                )

                out_final = self.process_output_for_loss(out)

                # The output should be single-sample and shouldn't have any padding.
                assert out_final.shape[0] == 1
                assert tl.eq_indicator(out, self.params.padding_indicator).sum().item() == 0

            list_counterfactual_predictions.append(out_final)

        data_historic_temporal_targets = data.temporal_targets[sample_index]
        if TYPE_CHECKING:
            assert isinstance(data_historic_temporal_targets, TimeSeries)
        list_ts = to_counterfactual_predictions(
            list_counterfactual_predictions, data_historic_temporal_targets, horizon
        )
        return list_ts


class CRNRegressor(CRNTreatmentEffectsModelBase):
    requirements: r.Requirements = r.Requirements(
        dataset_requirements=r.DatasetRequirements(
            temporal_covariates_value_type=r.DataValueOpts.NUMERIC,
            temporal_targets_value_type=r.DataValueOpts.NUMERIC_CATEGORICAL,
            temporal_treatments_value_type=r.DataValueOpts.NUMERIC_BINARY,
            static_covariates_value_type=r.DataValueOpts.NUMERIC,
            requires_no_missing_data=True,
        ),
        prediction_requirements=r.PredictionRequirements(
            target_data_structure=r.DataStructureOpts.TIME_SERIES,
            horizon_type=r.HorizonOpts.TIME_INDEX,
            min_timesteps_target_when_fit=6,
            min_timesteps_target_when_predict=3,
        ),
        treatment_effects_requirements=r.TreatmentEffectsRequirements(
            treatment_data_structure=r.DataStructureOpts.TIME_SERIES,
            min_timesteps_treatment_when_fit=6,
            min_timesteps_treatment_when_predict=3,
            min_timesteps_treatment_when_predict_counterfactual=3,
        ),
    )
    DEFAULT_PARAMS: TDefaultParams = _DefaultParams()

    def process_output_for_loss(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        return output

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(loss_fn=nn.MSELoss(), params=params)


class CRNClassifier(CRNTreatmentEffectsModelBase):
    requirements: r.Requirements = r.Requirements(
        dataset_requirements=r.DatasetRequirements(
            temporal_covariates_value_type=r.DataValueOpts.NUMERIC,
            temporal_targets_value_type=r.DataValueOpts.NUMERIC_CATEGORICAL,
            temporal_treatments_value_type=r.DataValueOpts.NUMERIC_BINARY,
            static_covariates_value_type=r.DataValueOpts.NUMERIC,
            requires_no_missing_data=True,
        ),
        prediction_requirements=r.PredictionRequirements(
            target_data_structure=r.DataStructureOpts.TIME_SERIES,
            horizon_type=r.HorizonOpts.TIME_INDEX,
            min_timesteps_target_when_fit=6,
            min_timesteps_target_when_predict=3,
        ),
        treatment_effects_requirements=r.TreatmentEffectsRequirements(
            treatment_data_structure=r.DataStructureOpts.TIME_SERIES,
            min_timesteps_treatment_when_fit=6,
            min_timesteps_treatment_when_predict=3,
            min_timesteps_treatment_when_predict_counterfactual=3,
        ),
    )
    DEFAULT_PARAMS: TDefaultParams = _DefaultParams()

    def process_output_for_loss(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.softmax(output)

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(loss_fn=nn.CrossEntropyLoss(), params=params)
        self.softmax = nn.Softmax(dim=-1)
