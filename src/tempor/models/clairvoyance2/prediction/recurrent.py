from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..components.torch.common import OPTIM_MAP
from ..components.torch.interfaces import (
    CustomizableLossMixin,
    OrganizedModule,
    OrganizedPredictorModuleMixin,
    SavableTorchModelMixin,
)
from ..components.torch.rnn import RecurrentFFNet, mask_and_reshape
from ..data import DEFAULT_PADDING_INDICATOR, Dataset, TimeSeriesSamples
from ..interface import Horizon, NStepAheadHorizon, PredictorModel, TDefaultParams, TParams
from ..interface import requirements as r
from ..utils import tensor_like as tl
from ..utils.array_manipulation import compute_deltas, n_step_shifted
from ..utils.dev import NEEDED

_DEBUG = False


class _DefaultParams(NamedTuple):
    rnn_type: str = "LSTM"
    hidden_size: int = 100
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0.0
    bidirectional: bool = False
    nonlinearity: Optional[str] = None
    proj_size: Optional[int] = None
    max_len: Optional[int] = None
    optimizer_str: str = "Adam"
    optimizer_kwargs: Mapping[str, Any] = dict(lr=0.01, weight_decay=1e-5)
    batch_size: int = 32
    epochs: int = 100
    use_past_targets: bool = True
    ff_hidden_dims: Sequence[int] = []
    ff_out_activation: Optional[str] = None
    padding_indicator: float = DEFAULT_PADDING_INDICATOR


class RecurrentNetNStepAheadPredictorBase(
    CustomizableLossMixin, SavableTorchModelMixin, PredictorModel, OrganizedPredictorModuleMixin, OrganizedModule
):
    requirements: r.Requirements
    DEFAULT_PARAMS: TDefaultParams

    def __init__(self, loss_fn: nn.Module, params: Optional[TParams] = None) -> None:
        PredictorModel.__init__(self, params)
        OrganizedModule.__init__(self)
        CustomizableLossMixin.__init__(self, loss_fn=loss_fn)

        # Decoder RNN and corresponding predictor FF NN:
        self.rnn_ff: Optional[RecurrentFFNet] = NEEDED

        # Torch necessities:
        self.optim: Optional[torch.optim.Optimizer] = NEEDED

    def _init_submodules(self) -> None:
        self.rnn_ff = RecurrentFFNet(
            rnn_type=self.params.rnn_type,
            input_size=self.inferred_params.rnn_input_size,
            hidden_size=self.params.hidden_size,
            nonlinearity=self.params.nonlinearity,
            num_layers=self.params.num_layers,
            bias=self.params.bias,
            dropout=self.params.dropout,
            bidirectional=self.params.bidirectional,
            proj_size=self.params.proj_size,
            ff_out_size=self.inferred_params.ff_output_size,
            ff_hidden_dims=self.params.ff_hidden_dims,
            ff_out_activation=self.params.ff_out_activation,
            ff_hidden_activations="ReLU",
        )

    def _init_inferred_params(self, data: Dataset, **kwargs) -> None:
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None

        # Infer some parameters from data.
        self.inferred_params.rnn_input_size = data.temporal_covariates.n_features + 1  # + 1 for time deltas.
        if self.params.use_past_targets:
            self.inferred_params.rnn_input_size += data.temporal_targets.n_features
        self.inferred_params.ff_output_size = data.temporal_targets.n_features

        if _DEBUG is True:  # pragma: no cover
            print("Inferred rnn_input_size:", self.inferred_params.rnn_input_size)
            print("Inferred ff_input_size:", self.inferred_params.ff_input_size)
            print("Inferred ff_output_size:", self.inferred_params.ff_output_size)

        # Inferred batch size:
        self.inferred_params.batch_size = min(self.params.batch_size, data.n_samples)

    def _init_optimizers(self):
        # Initialize optimizer.
        self.optim = OPTIM_MAP[self.params.optimizer_str](
            params=self.rnn_ff.parameters(),
            **self.params.optimizer_kwargs,
        )

    def _prep_torch_tensors(self, data: Dataset, horizon: NStepAheadHorizon, shift: bool):
        if TYPE_CHECKING:
            assert data.temporal_targets is not None
        t_cov = data.temporal_covariates.to_torch_tensor(
            padding_indicator=self.params.padding_indicator,
            max_len=self.params.max_len,
            dtype=self.dtype,
            device=self.device,
        )
        t_cov_ti = data.temporal_covariates.to_torch_tensor_time_index(
            padding_indicator=self.params.padding_indicator,
            max_len=self.params.max_len,
            dtype=self.dtype,
            device=self.device,
        )
        time_deltas = compute_deltas(t_cov_ti, padding_indicator=self.params.padding_indicator)
        t_cov = torch.cat([t_cov, time_deltas], dim=-1)
        t_targ = data.temporal_targets.to_torch_tensor(
            padding_indicator=self.params.padding_indicator,
            max_len=self.params.max_len,
            dtype=self.dtype,
            device=self.device,
        )

        if self.params.use_past_targets:
            t_cov = torch.cat([t_targ, t_cov], dim=-1)
        if shift:
            t_targ, t_cov = n_step_shifted(
                t_targ,
                t_cov,
                horizon.n_step,
                padding_indicator=self.params.padding_indicator,
            )

        return t_cov, t_targ

    def _prep_data_for_fit(
        self, data: Dataset, horizon: Optional[Horizon] = NEEDED, **kwargs
    ) -> Tuple[DataLoader, ...]:
        assert horizon is not None
        assert isinstance(horizon, NStepAheadHorizon)
        t_cov, t_targ = self._prep_torch_tensors(data, horizon, shift=True)
        dataloader = DataLoader(TensorDataset(t_cov, t_targ), batch_size=self.inferred_params.batch_size, shuffle=True)
        return (dataloader,)

    def _prep_data_for_predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> Tuple[torch.Tensor, ...]:
        assert isinstance(horizon, NStepAheadHorizon)
        t_cov, _ = self._prep_torch_tensors(data, horizon, shift=False)
        return (t_cov,)

    def _prep_submodules_for_fit(self) -> None:
        assert self.rnn_ff is not None
        self.rnn_ff.to(self.device, dtype=self.dtype)
        self.rnn_ff.train()

    def _prep_submodules_for_predict(self) -> None:
        assert self.rnn_ff is not None
        self.rnn_ff.to(self.device, dtype=self.dtype)
        self.rnn_ff.eval()

    def _fit(
        self, data: Dataset, horizon: Optional[Horizon] = NEEDED, **kwargs
    ) -> "RecurrentNetNStepAheadPredictorBase":
        self.set_attributes_from_kwargs(**kwargs)
        dataloader, *_ = self.prep_fit(data=data, horizon=horizon)

        if TYPE_CHECKING:
            assert self.rnn_ff is not None and self.optim is not None

        for epoch_idx in range(self.params.epochs):
            n_samples = 0
            epoch_loss = 0.0
            for batch_idx, (t_cov, t_targ) in enumerate(dataloader):
                current_batch_size = t_cov.shape[0]
                n_samples += current_batch_size

                if _DEBUG is True:  # pragma: no cover
                    print("t_targ.shape", t_targ.shape)
                    print("x.shape", t_cov.shape)
                    print("y.shape", t_targ.shape)

                out, _, _ = self.rnn_ff(t_cov, h=None, padding_indicator=self.params.padding_indicator)

                if _DEBUG is True:  # pragma: no cover
                    print("out.shape", out.shape)

                not_padding = ~tl.eq_indicator(t_targ, self.params.padding_indicator)
                if TYPE_CHECKING:
                    assert isinstance(not_padding, torch.BoolTensor)
                out = mask_and_reshape(mask_selector=not_padding, tensor=out)
                t_targ = mask_and_reshape(mask_selector=not_padding, tensor=t_targ)

                out = self.process_output_for_loss(out)
                loss = self.loss_fn(out, t_targ)

                # Optimization:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item() * n_samples

                if _DEBUG is True:  # pragma: no cover
                    print(f"{batch_idx}: {loss.item()}")

            epoch_loss /= n_samples
            print(f"Epoch: {epoch_idx}, Loss: {epoch_loss}")

        return self

    def _predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> TimeSeriesSamples:
        self.set_attributes_from_kwargs(**kwargs)
        t_cov, *_ = self.prep_predict(data=data, horizon=horizon)
        if TYPE_CHECKING:
            assert self.rnn_ff is not None
            assert horizon is not None
            assert isinstance(horizon, NStepAheadHorizon)
            assert data.temporal_targets is not None

        with torch.no_grad():
            # NOTE: The N-step ahead output will have the n. timesteps == original data temporal targets n. timesteps.
            # But the prediction values correspond to the n step shifted targets.
            out, _, _ = self.rnn_ff(t_cov, h=None, padding_indicator=self.params.padding_indicator)
            out_final = self.process_output_for_loss(out)
            out_final[tl.eq_indicator(out, self.params.padding_indicator)] = self.params.padding_indicator

        result = out_final.detach().cpu().numpy()

        prediction = data.temporal_targets.copy()
        prediction.update_from_array_n_step_ahead(
            update_array_sequence=result, n_step=horizon.n_step, padding_indicator=self.params.padding_indicator
        )
        return prediction


class RNNRegressor(RecurrentNetNStepAheadPredictorBase):
    requirements: r.Requirements = r.Requirements(
        dataset_requirements=r.DatasetRequirements(
            temporal_covariates_value_type=r.DataValueOpts.NUMERIC,
            temporal_targets_value_type=r.DataValueOpts.NUMERIC,
            static_covariates_value_type=r.DataValueOpts.NUMERIC,
            requires_no_missing_data=True,
        ),
        prediction_requirements=r.PredictionRequirements(
            target_data_structure=r.DataStructureOpts.TIME_SERIES,
            horizon_type=r.HorizonOpts.N_STEP_AHEAD,
        ),
    )
    DEFAULT_PARAMS: TDefaultParams = _DefaultParams()

    def process_output_for_loss(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        return output

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(loss_fn=nn.MSELoss(), params=params)


class RNNClassifier(RecurrentNetNStepAheadPredictorBase):
    requirements: r.Requirements = r.Requirements(
        dataset_requirements=r.DatasetRequirements(
            temporal_covariates_value_type=r.DataValueOpts.NUMERIC,
            temporal_targets_value_type=r.DataValueOpts.NUMERIC_CATEGORICAL,
            static_covariates_value_type=r.DataValueOpts.NUMERIC,
            requires_no_missing_data=True,
        ),
        prediction_requirements=r.PredictionRequirements(
            target_data_structure=r.DataStructureOpts.TIME_SERIES,
            horizon_type=r.HorizonOpts.N_STEP_AHEAD,
        ),
    )
    DEFAULT_PARAMS: TDefaultParams = _DefaultParams()

    def process_output_for_loss(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.softmax(output)

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(loss_fn=nn.CrossEntropyLoss(), params=params)
        self.softmax = nn.Softmax(dim=-1)
