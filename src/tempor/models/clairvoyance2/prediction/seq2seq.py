# mypy: ignore-errors

from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..components.torch.common import OPTIM_MAP
from ..components.torch.ffnn import FeedForwardNet
from ..components.torch.interfaces import (
    CustomizableLossMixin,
    OrganizedModule,
    OrganizedPredictorModuleMixin,
    SavableTorchModelMixin,
)
from ..components.torch.rnn import RecurrentFFNet, RNNHidden, mask_and_reshape
from ..data import DEFAULT_PADDING_INDICATOR, Dataset, StaticSamples, TimeSeriesSamples
from ..data.utils import split_time_series, time_index_utils
from ..interface import Horizon, PredictorModel, TDefaultParams, TimeIndexHorizon, TParams
from ..interface import requirements as r
from ..utils import tensor_like as tl
from ..utils.array_manipulation import compute_deltas, n_step_shifted
from ..utils.dev import NEEDED

_DEBUG = False


# TODO: For clarity, get rid of TEncodedRepresentation and always use RNNHidden?
TEncodedRepresentation = Tuple[torch.Tensor, Optional[torch.Tensor]]


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
    # Misc:
    max_len: Optional[int] = None
    optimizer_str: str = "Adam"
    optimizer_kwargs: Mapping[str, Any] = dict(lr=0.01, weight_decay=1e-5)
    batch_size: int = 32
    epochs: int = 100
    padding_indicator: float = DEFAULT_PADDING_INDICATOR


class Seq2SeqCRNStylePredictorBase(
    CustomizableLossMixin, SavableTorchModelMixin, PredictorModel, OrganizedPredictorModuleMixin, OrganizedModule
):
    requirements: r.Requirements
    DEFAULT_PARAMS: TDefaultParams

    def __init__(self, loss_fn: nn.Module, params: Optional[TParams] = None) -> None:
        PredictorModel.__init__(self, params)
        OrganizedModule.__init__(self)
        CustomizableLossMixin.__init__(self, loss_fn=loss_fn)

        # Decoder RNN and corresponding predictor FF NN, and optimizer:
        self.encoder: Optional[RecurrentFFNet] = NEEDED
        self.encoder_optim: Optional[torch.optim.Optimizer] = NEEDED

        # Decoder RNN and corresponding predictor FF NN, and optimizer:
        self.decoder: Optional[RecurrentFFNet] = NEEDED
        self.decoder_optim: Optional[torch.optim.Optimizer] = NEEDED

        # Adapter.
        self.adapter: Optional[FeedForwardNet] = NEEDED

        # Helpers.
        self.encoder_output_and_h_dim: Tuple[int, int, int] = (-1, -1, -1)  # To be set.
        self.decoder_output_and_h_dim: Tuple[int, int, int] = (-1, -1, -1)  # To be set.

    def _init_submodules_encoder_decoder(self) -> None:
        # Initialize Encoder models:
        self.encoder = RecurrentFFNet(
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
            ff_hidden_dims=self.params.predictor_hidden_dims,
            ff_out_activation=self.params.predictor_out_activation,
            ff_hidden_activations="ReLU",
        )

        # Initialize Decoder models:
        self.decoder = RecurrentFFNet(
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
            ff_hidden_dims=self.params.predictor_hidden_dims,
            ff_out_activation=self.params.predictor_out_activation,
            ff_hidden_activations="ReLU",
        )

    def _init_submodules_adapter(self) -> None:
        if TYPE_CHECKING:
            assert self.encoder is not None and self.decoder is not None

        # Initialize Adapter model:
        _, (d_num_layers, h_out, h_cell) = self.encoder.rnn.get_output_and_h_dim()
        self.encoder_output_and_h_dim = (d_num_layers, h_out, h_cell)
        self.inferred_params.adapter_input_size = d_num_layers * h_out + d_num_layers * h_cell
        _, (d_num_layers, h_out, h_cell) = self.decoder.rnn.get_output_and_h_dim()
        self.decoder_output_and_h_dim = (d_num_layers, h_out, h_cell)
        self.inferred_params.adapter_output_size = d_num_layers * h_out + d_num_layers * h_cell
        if _DEBUG:
            print("self.inferred_params.adapter_input_size", self.inferred_params.adapter_input_size)
        self.adapter = FeedForwardNet(
            in_dim=self.inferred_params.adapter_input_size,
            out_dim=self.inferred_params.adapter_output_size,
            hidden_dims=self.params.adapter_hidden_dims,
            out_activation=self.params.adapter_out_activation,
            hidden_activations="ReLU",
        )

    def _init_submodules(self) -> None:
        self._init_submodules_encoder_decoder()
        self._init_submodules_adapter()

    def _init_inferred_params(self, data: Dataset, **kwargs) -> None:
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None

        # Initialize the helper attributes.
        self.inferred_params.encoder_input_size = data.temporal_covariates.n_features + 1  # + 1 for time deltas.
        if data.static_covariates is not None:
            self.inferred_params.encoder_input_size += data.static_covariates.n_features
        self.inferred_params.decoder_input_size = data.temporal_targets.n_features + 1  # + 1 for time deltas.
        if data.static_covariates is not None:
            self.inferred_params.decoder_input_size += data.static_covariates.n_features
        self.inferred_params.encoder_input_size += data.temporal_targets.n_features
        self.inferred_params.encoder_predictor_output_size = data.temporal_targets.n_features
        self.inferred_params.decoder_predictor_output_size = data.temporal_targets.n_features

        # Inferred batch size:
        self.inferred_params.encoder_batch_size = min(self.params.batch_size, data.n_samples)
        self.inferred_params.decoder_batch_size = NEEDED  # This is set later.

    def _init_optimizers(self):
        # Initialize optimizers.
        self.encoder_optim = OPTIM_MAP[self.params.optimizer_str](
            params=self.encoder.parameters(),  # type: ignore
            **self.params.optimizer_kwargs,
        )
        self.decoder_optim = OPTIM_MAP[self.params.optimizer_str](
            params=[*self.adapter.parameters(), *self.decoder.parameters()],  # type: ignore
            **self.params.optimizer_kwargs,
        )

    def _prep_torch_tensors_encoder(self, data: Dataset, shift_targ_cov: bool):
        if TYPE_CHECKING:
            assert data.temporal_targets is not None
        t_cov = data.temporal_covariates.to_torch_tensor(
            padding_indicator=self.params.padding_indicator,
            max_len=self.params.max_len,
            dtype=self.dtype,
            device=self.device,
        )
        n_samples, n_timesteps = t_cov.shape[0], t_cov.shape[1]
        t_cov_ti = data.temporal_covariates.to_torch_tensor_time_index(
            padding_indicator=self.params.padding_indicator,
            max_len=self.params.max_len,
            dtype=self.dtype,
            device=self.device,
        )
        time_deltas = compute_deltas(t_cov_ti, padding_indicator=self.params.padding_indicator)
        t_cov = torch.cat([t_cov, time_deltas], dim=-1)

        t_targ = data.temporal_targets.to_torch_tensor(dtype=self.dtype, device=self.device)
        t_cov = torch.cat([t_targ, t_cov], dim=-1)

        if data.static_covariates is not None:
            s_cov = data.static_covariates.to_torch_tensor(dtype=self.dtype, device=self.device)
            s_cov_repeated = s_cov.repeat(repeats=(n_timesteps, 1, 1)).reshape([n_samples, n_timesteps, -1])
            t_cov = torch.cat([t_cov, s_cov_repeated], dim=-1)

        if shift_targ_cov:
            t_targ, t_cov = n_step_shifted(
                t_targ,
                t_cov,
                n_step=1,
                padding_indicator=self.params.padding_indicator,
            )

        return t_cov, t_targ

    def _reshape_h_sample_dim_0(self, encoded_representations: TEncodedRepresentation):
        # Reshape such that the 0th dimension is the sample (batch) dimension.
        h, c = encoded_representations
        h = h.reshape(h.shape[1], h.shape[0], h.shape[2])
        h = h.to(device=self.device)
        if c is not None:
            c = c.reshape(c.shape[1], c.shape[0], c.shape[2])
            c = c.to(device=self.device)
        if c is None:
            c = torch.full_like(h, fill_value=torch.nan)
        return h, c

    def _concat_to_make_decoder_input(
        self, static_covariates: Optional[StaticSamples], t_targ: torch.Tensor, t_targ_td: torch.Tensor
    ):
        n_samples, n_timesteps = t_targ_td.shape[0], t_targ_td.shape[1]
        decoder_input_to_cat = [t_targ, t_targ_td]

        if static_covariates is not None:
            s_cov = static_covariates.to_torch_tensor(dtype=self.dtype, device=self.device)
            s_cov_repeated = s_cov.repeat(repeats=(n_timesteps, 1, 1)).reshape([n_samples, n_timesteps, -1])
            decoder_input_to_cat.append(s_cov_repeated)

        decoder_input = torch.cat(decoder_input_to_cat, dim=-1)
        return decoder_input

    def _prep_torch_tensors_decoder(self, data: Dataset):
        if TYPE_CHECKING:
            assert data.temporal_targets is not None

        t_targ = data.temporal_targets.to_torch_tensor(
            padding_indicator=self.params.padding_indicator,
            max_len=self.params.max_len,
            dtype=self.dtype,
            device=self.device,
        )
        t_targ_ti = data.temporal_targets.to_torch_tensor_time_index(
            padding_indicator=self.params.padding_indicator,
            max_len=self.params.max_len,
            dtype=self.dtype,
            device=self.device,
        )
        t_targ_td = compute_deltas(t_targ_ti, padding_indicator=self.params.padding_indicator)

        decoder_input = self._concat_to_make_decoder_input(data.static_covariates, t_targ=t_targ, t_targ_td=t_targ_td)

        t_targ, decoder_input = n_step_shifted(
            t_targ,
            decoder_input,
            n_step=1,
            padding_indicator=self.params.padding_indicator,
        )

        return t_targ, decoder_input

    def _prep_torch_tensors_decoder_inference(
        self,
        data: Dataset,
        horizon: TimeIndexHorizon,
    ):
        if TYPE_CHECKING:
            assert data.temporal_targets is not None

        t_targ_ti = horizon.to_torch_time_series(
            padding_indicator=self.params.padding_indicator,
            max_len=None,
            dtype=self.dtype,
            device=self.device,
        )
        t_targ_td = compute_deltas(t_targ_ti, padding_indicator=self.params.padding_indicator)

        # Prepare t_targ:
        n_samples = data.n_samples
        n_timesteps = t_targ_td.shape[1]
        n_target_features = data.temporal_targets.n_features
        # Set up the right shape, but fill with nan, as these values should not be used.
        t_targ = torch.full(size=(n_samples, n_timesteps, n_target_features), fill_value=torch.nan, device=self.device)
        # The only values that need filling are the 0th timestep:
        # take the target(s) just before the 0th step defined in horizon time index.
        ts_targ = time_index_utils.time_series_samples.take_one_before_start(
            data.temporal_targets, horizon, inplace=False
        )
        t_targ_0 = ts_targ.to_torch_tensor(  # type: ignore
            padding_indicator=self.params.padding_indicator,
            max_len=1,
            dtype=self.dtype,
            device=self.device,
        )
        t_targ[:, [0], :] = t_targ_0

        decoder_input = self._concat_to_make_decoder_input(data.static_covariates, t_targ=t_targ, t_targ_td=t_targ_td)

        return decoder_input

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

        t_cov_to_encode, _ = self._prep_torch_tensors_encoder(data_pre, shift_targ_cov=False)

        decoder_tensors = self._prep_torch_tensors_decoder(data_post)

        return (*encoder_tensors, t_cov_to_encode, *decoder_tensors)

    def _prep_data_for_predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> Tuple[torch.Tensor, ...]:
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None
        assert isinstance(horizon, TimeIndexHorizon)

        # Make sure to not use "future" values for prediction.
        data_encode = time_index_utils.dataset.take_temporal_data_before_start(data, horizon, inplace=False)
        if TYPE_CHECKING:
            assert data_encode is not None

        t_cov_to_encode, _ = self._prep_torch_tensors_encoder(data_encode, shift_targ_cov=False)
        encoded_representations = self._get_encoder_representation(t_cov_to_encode)
        h, c = self._reshape_h_sample_dim_0(encoded_representations)

        decoder_input = self._prep_torch_tensors_decoder_inference(data, horizon)
        assert h.shape[0] == decoder_input.shape[0]

        return h, c, decoder_input

    def _prep_submodules_for_fit(self) -> None:
        assert self.encoder is not None and self.decoder is not None and self.adapter is not None
        self.encoder.to(self.device, dtype=self.dtype)
        self.adapter.to(self.device, dtype=self.dtype)
        self.decoder.to(self.device, dtype=self.dtype)
        self.encoder.train()
        self.adapter.train()
        self.decoder.train()

    def _prep_submodules_for_predict(self) -> None:
        assert self.encoder is not None and self.decoder is not None and self.adapter is not None
        self.encoder.to(self.device, dtype=self.dtype)
        self.adapter.to(self.device, dtype=self.dtype)
        self.decoder.to(self.device, dtype=self.dtype)
        self.encoder.eval()
        self.adapter.eval()
        self.decoder.eval()

    def _prep_submodules_for_predict_counterfactuals(self) -> None:
        self._prep_submodules_for_predict()

    def _train_encoder(self, encoder_tensors: Tuple) -> None:
        # 1 step ahead training.
        if TYPE_CHECKING:
            assert self.encoder is not None
            assert self.encoder_optim is not None

        dataloader = DataLoader(
            TensorDataset(*encoder_tensors), batch_size=self.inferred_params.encoder_batch_size, shuffle=True
        )

        for epoch_idx in range(self.params.epochs):
            n_samples_cumul = 0
            epoch_loss = 0.0
            for _, (t_cov, t_targ) in enumerate(dataloader):
                current_batch_size = t_cov.shape[0]
                n_samples_cumul += current_batch_size

                out, _, _ = self.encoder(t_cov, h=None, padding_indicator=self.params.padding_indicator)

                not_padding = ~tl.eq_indicator(t_targ, self.params.padding_indicator)
                if TYPE_CHECKING:
                    assert isinstance(not_padding, torch.BoolTensor)
                out = mask_and_reshape(mask_selector=not_padding, tensor=out)
                t_targ = mask_and_reshape(mask_selector=not_padding, tensor=t_targ)

                out = self.process_output_for_loss(out)
                loss = self.loss_fn(out, t_targ)

                # Optimization:
                self.encoder_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()

                epoch_loss += loss.item() * current_batch_size

            epoch_loss /= n_samples_cumul
            print(f"Epoch: {epoch_idx}, Loss: {epoch_loss}")

    def _get_encoder_representation(  # pylint: disable=unused-argument
        self, t_cov: torch.Tensor, **kwargs
    ) -> TEncodedRepresentation:
        # 2. Get the encoded representations.
        if TYPE_CHECKING:
            assert self.encoder is not None

        # Not sure this is needed here, but just in case:
        self.encoder.eval()

        is_lstm = self.params.encoder_rnn_type == "LSTM"
        with torch.no_grad():
            _, _, h = self.encoder(t_cov, h=None, padding_indicator=self.params.padding_indicator)
            h, c = h if is_lstm else (h, None)

        return h, c

    def _shape_h_for_adapter(
        self, encoded_representations: TEncodedRepresentation, n_samples: int, h_dim_info: Tuple[int, int, int]
    ) -> torch.Tensor:
        if _DEBUG is True:  # pragma: no cover
            print("_shape_for_adapter()...")
        h, c = encoded_representations
        (d_num_layers, h_out, h_cell) = h_dim_info
        if h_cell > 0:  # LSTM.
            assert c is not None
            h = torch.cat([h, c], dim=-1)
        if _DEBUG is True:  # pragma: no cover
            print("h before reshaping:", h.shape)
        combined = d_num_layers * (h_out + h_cell)
        h_for_adapter = h.reshape(n_samples, combined)
        if _DEBUG is True:  # pragma: no cover
            print("h after reshaping:", h_for_adapter.shape)
        return h_for_adapter

    def _shape_h_back_after_adapter(
        self, h_after_adapter: torch.Tensor, n_samples: int, h_dim_info: Tuple[int, int, int]
    ) -> RNNHidden:
        if _DEBUG is True:  # pragma: no cover
            print("_shape_back_after_adapter()...")
            print("h_adapter_output initial shape:", h_after_adapter.shape)
        (d_num_layers, h_out, h_cell) = h_dim_info
        h_return: RNNHidden = h_after_adapter.reshape(d_num_layers, n_samples, h_out + h_cell)
        if _DEBUG is True:  # pragma: no cover
            print("h_adapter_output after reshaping:", h_after_adapter.shape)
        if h_cell > 0:  # LSTM.
            assert isinstance(h_return, torch.Tensor)
            h_return = h_return[:, :, :h_out]
            c_adapted = h_return[:, :, -h_cell:]
            h_return = (h_return, c_adapted)
            if _DEBUG is True:  # pragma: no cover
                print("h_adapter_output after splitting:", h_after_adapter[0].shape)
                print("h_adapter_output after splitting:", h_after_adapter[1].shape)
        return h_return

    def _pass_h_through_adapter(self, h, c):
        n_samples = h.shape[0]  # NOTE: Expects h/c to have sample dim as 0th dimension.
        h_adapted = self._shape_h_for_adapter(
            encoded_representations=(h, c), n_samples=n_samples, h_dim_info=self.encoder_output_and_h_dim
        )
        h_adapted = self.adapter(h_adapted)  # type: ignore
        h_adapted = self._shape_h_back_after_adapter(
            h_after_adapter=h_adapted,
            n_samples=n_samples,
            h_dim_info=self.decoder_output_and_h_dim,
        )
        return h_adapted

    def _train_decoder(self, encoded_representations: TEncodedRepresentation, decoder_tensors: Tuple) -> None:
        if TYPE_CHECKING:
            assert self.encoder is not None and self.decoder is not None
            assert self.decoder_optim is not None
            assert self.adapter is not None

        (t_targ, decoder_input) = decoder_tensors
        h, c = self._reshape_h_sample_dim_0(encoded_representations)
        assert h.shape[0] == decoder_input.shape[0]

        dataloader = DataLoader(
            TensorDataset(h, c, t_targ, decoder_input),
            batch_size=self.inferred_params.decoder_batch_size,
            shuffle=True,
        )

        for epoch_idx in range(self.params.epochs):
            n_samples_cumul = 0
            epoch_loss = 0.0
            for _, (h, c, t_targ, decoder_input) in enumerate(dataloader):
                current_batch_size = t_targ.shape[0]
                n_samples_cumul += current_batch_size

                # Pass encoded representations through the adapter.
                h_adapter_out = self._pass_h_through_adapter(h, c)

                out, _, _ = self.decoder(
                    decoder_input, h=h_adapter_out, padding_indicator=self.params.padding_indicator
                )

                not_padding = ~tl.eq_indicator(t_targ, self.params.padding_indicator)
                if TYPE_CHECKING:
                    assert isinstance(not_padding, torch.BoolTensor)
                out = mask_and_reshape(mask_selector=not_padding, tensor=out)
                t_targ = mask_and_reshape(mask_selector=not_padding, tensor=t_targ)

                out = self.process_output_for_loss(out)
                loss = self.loss_fn(out, t_targ)

                # Optimization:
                self.decoder_optim.zero_grad()
                loss.backward()
                self.decoder_optim.step()

                epoch_loss += loss.item() * current_batch_size

            epoch_loss /= n_samples_cumul
            print(f"Epoch: {epoch_idx}, Loss: {epoch_loss}")

    def _fit(self, data: Dataset, horizon: Horizon = None, **kwargs) -> "Seq2SeqCRNStylePredictorBase":  # type: ignore
        self.set_attributes_from_kwargs(**kwargs)

        # Ensure there are at least 2 timesteps in the posterior part of TimeSeries after the split
        # (as we need to shift targets by 1).
        encoder_t_cov, encoder_t_targ, t_cov_to_encode, decoder_t_targ, decoder_input = self.prep_fit(
            data=data, min_pre_len=1, min_post_len=2, repeat_last_pre_step=True
        )

        # Run the training stages.
        print("=== Training stage: 1. Train encoder ===")
        self._train_encoder(encoder_tensors=(encoder_t_cov, encoder_t_targ))

        print("=== Training stage: 2. Train decoder ===")
        if TYPE_CHECKING:
            assert isinstance(t_cov_to_encode, torch.Tensor)
        encoded_representations = self._get_encoder_representation(t_cov_to_encode)
        self._train_decoder(encoded_representations, decoder_tensors=(decoder_t_targ, decoder_input))

        return self

    def _predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> TimeSeriesSamples:
        self.set_attributes_from_kwargs(**kwargs)

        data = data.copy()
        if TYPE_CHECKING:
            assert self.decoder is not None
            assert data.temporal_targets is not None
            assert isinstance(horizon, TimeIndexHorizon)

        h, c, decoder_input = self.prep_predict(data, horizon=horizon)
        if TYPE_CHECKING:
            assert isinstance(h, torch.Tensor)
            assert isinstance(decoder_input, torch.Tensor)

        with torch.no_grad():
            h_adapter_out = self._pass_h_through_adapter(h, c)
            out = self.decoder.autoregress(
                decoder_input, h=h_adapter_out, padding_indicator=self.params.padding_indicator
            )

            out_final: Any = self.process_output_for_loss(out)
            out_final[tl.eq_indicator(out, self.params.padding_indicator)] = self.params.padding_indicator

        prediction = TimeSeriesSamples.new_empty_like(like=data.temporal_targets)
        prediction.update_from_sequence_of_arrays(
            out_final, time_index_sequence=horizon.time_index_sequence, padding_indicator=self.params.padding_indicator
        )
        return prediction


class Seq2SeqRegressor(Seq2SeqCRNStylePredictorBase):
    requirements: r.Requirements = r.Requirements(
        dataset_requirements=r.DatasetRequirements(
            temporal_covariates_value_type=r.DataValueOpts.NUMERIC,
            temporal_targets_value_type=r.DataValueOpts.NUMERIC,
            static_covariates_value_type=r.DataValueOpts.NUMERIC,
            requires_no_missing_data=True,
        ),
        prediction_requirements=r.PredictionRequirements(
            target_data_structure=r.DataStructureOpts.TIME_SERIES,
            horizon_type=r.HorizonOpts.TIME_INDEX,
            min_timesteps_target_when_fit=3,
            min_timesteps_target_when_predict=1,
        ),
    )
    DEFAULT_PARAMS: TDefaultParams = _DefaultParams()

    def process_output_for_loss(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        return output

    def __init__(self, params: Optional[TParams] = None) -> None:
        Seq2SeqCRNStylePredictorBase.__init__(self, loss_fn=nn.MSELoss(), params=params)


class Seq2SeqClassifier(Seq2SeqCRNStylePredictorBase):
    requirements: r.Requirements = r.Requirements(
        dataset_requirements=r.DatasetRequirements(
            temporal_covariates_value_type=r.DataValueOpts.NUMERIC,
            temporal_targets_value_type=r.DataValueOpts.NUMERIC_CATEGORICAL,
            static_covariates_value_type=r.DataValueOpts.NUMERIC,
            requires_no_missing_data=True,
        ),
        prediction_requirements=r.PredictionRequirements(
            target_data_structure=r.DataStructureOpts.TIME_SERIES,
            horizon_type=r.HorizonOpts.TIME_INDEX,
            min_timesteps_target_when_fit=3,
            min_timesteps_target_when_predict=1,
        ),
    )
    DEFAULT_PARAMS: TDefaultParams = _DefaultParams()

    def process_output_for_loss(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.softmax(output)

    def __init__(self, params: Optional[TParams] = None) -> None:
        Seq2SeqCRNStylePredictorBase.__init__(self, loss_fn=nn.CrossEntropyLoss(), params=params)
        self.softmax = nn.Softmax(dim=-1)
