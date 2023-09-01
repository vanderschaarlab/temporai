from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Literal, Self, get_args

from tempor.models import constants

from .mlp import MLP
from .transformer import TransformerModel
from .ts_model import TimeSeriesLayer

RnnMode = Literal[
    "GRU",
    "LSTM",
    "RNN",
    "Transformer",
]
OutputMode = Literal[
    "MLP",
    "LSTM",
    "GRU",
    "RNN",
    "Transformer",
    "TCN",
    "InceptionTime",
    "InceptionTimePlus",
    "ResCNN",
    "XCM",
]

rnn_modes = get_args(RnnMode)
output_modes = get_args(OutputMode)


def get_padded_features(
    x: Union[np.ndarray, List[np.ndarray]], pad_size: Optional[int] = None, fill: float = np.nan
) -> np.ndarray:
    """Helper function to pad variable length RNN inputs with nans."""
    if pad_size is None:
        pad_size = max([len(x_) for x_ in x])

    x_padded = []
    for i in range(len(x)):
        if pad_size == len(x[i]):
            x_padded.append(x[i].astype(float))
        elif pad_size > len(x[i]):
            pads = fill * np.ones((pad_size - len(x[i]),) + x[i].shape[1:])
            x_padded.append(np.concatenate([x[i], pads]).astype(float))
        else:
            x_padded.append(x[i][:pad_size].astype(float))

    return np.asarray(x_padded)


class DynamicDeepHitModel:
    """This implementation considers that the last event happen at the same time for each patient.
    The CIF is therefore simplified.
    """

    def __init__(
        self,
        split: int = 100,
        n_layers_hidden: int = 2,
        n_units_hidden: int = 100,
        rnn_mode: str = "LSTM",
        dropout: float = 0.1,
        alpha: float = 0.1,
        beta: float = 0.1,
        sigma: float = 0.1,
        patience: int = 20,
        lr: float = 1e-3,
        batch_size: int = 100,
        n_iter: int = 1000,
        device: Any = constants.DEVICE,
        val_size: float = 0.1,
        random_state: int = 0,
        clipping_value: int = 1,
        output_mode: str = "MLP",
    ) -> None:
        self.split = split
        self.split_time = None

        self.pad_size = 0

        self.layers_rnn = n_layers_hidden
        self.hidden_rnn = n_units_hidden
        self.rnn_type = rnn_mode

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dropout = dropout
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_size = val_size
        self.clipping_value = clipping_value

        self.patience = patience
        self.random_state = random_state
        self.output_type = output_mode

        self.model: Optional[DynamicDeepHitLayers] = None

    def _setup_model(self, inputdim: int, seqlen: int, risks: int) -> "DynamicDeepHitLayers":
        return (
            DynamicDeepHitLayers(
                inputdim,
                seqlen,
                self.split,
                self.layers_rnn,
                self.hidden_rnn,
                rnn_type=self.rnn_type,
                dropout=self.dropout,
                risks=risks,
                device=self.device,
                output_type=self.output_type,
            )
            .float()
            .to(self.device)
        )

    def fit(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
    ) -> Self:
        discretized_t, self.split_time = self.discretize(t, self.split, self.split_time)
        processed_data = self._preprocess_training_data(x, discretized_t, e)
        x_train, t_train, e_train, x_val, t_val, e_val = processed_data
        inputdim = x_train.shape[-1]
        seqlen = x_train.shape[-2]

        maxrisk = int(np.nanmax(e_train.cpu().numpy()))

        self.model = self._setup_model(inputdim, seqlen, risks=maxrisk)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        patience, old_loss = 0, np.inf
        nbatches = int(x_train.shape[0] / self.batch_size) + 1
        valbatches = int(x_val.shape[0] / self.batch_size) + 1

        best_param = deepcopy(self.model.state_dict())

        for i in range(self.n_iter):  # pylint: disable=unused-variable
            self.model.train()
            for j in range(nbatches):
                xb = x_train[j * self.batch_size : (j + 1) * self.batch_size]
                tb = t_train[j * self.batch_size : (j + 1) * self.batch_size]
                eb = e_train[j * self.batch_size : (j + 1) * self.batch_size]

                if xb.shape[0] == 0:  # pragma: no cover
                    continue

                optimizer.zero_grad()
                loss = self.total_loss(xb, tb, eb)
                loss.backward()

                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(  # pyright: ignore [reportPrivateImportUsage]
                        self.model.parameters(),
                        self.clipping_value,
                    )

                optimizer.step()

            self.model.eval()
            valid_loss: Any = 0.0
            for j in range(valbatches):
                xb = x_val[j * self.batch_size : (j + 1) * self.batch_size]
                tb = t_val[j * self.batch_size : (j + 1) * self.batch_size]
                eb = e_val[j * self.batch_size : (j + 1) * self.batch_size]

                if xb.shape[0] == 0:  # pragma: no cover
                    continue

                valid_loss += self.total_loss(xb, tb, eb)

            if torch.isnan(valid_loss):  # pragma: no cover
                raise RuntimeError("NaNs detected in the total loss")

            valid_loss = valid_loss.item()

            if valid_loss < old_loss:
                patience = 0
                old_loss = valid_loss
                best_param = deepcopy(self.model.state_dict())
            else:
                patience += 1

            if patience == self.patience:
                break

        self.model.load_state_dict(best_param)
        self.model.eval()

        return self

    def discretize(self, t: Union[np.ndarray, List[np.ndarray]], split: int, split_time: Optional[int] = None) -> Tuple:
        """Discretize the survival horizon.

        Args:
            t (List of Array): Time of events
            split (int): Number of bins
            split_time (List, optional): List of bins (must be same length than split). Defaults to None.

        Returns:
            List of Array: Discretized events time
        """
        if split_time is None:
            _, split_time = np.histogram(t, split - 1)  # type: ignore
        t_discretized = np.array(
            [np.digitize(t_, split_time, right=True) - 1 for t_ in t], dtype=object  # type: ignore
        )
        return t_discretized, split_time

    def _preprocess_test_data(self, x: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        data = torch.from_numpy(get_padded_features(x, pad_size=self.pad_size)).float().to(self.device)
        return data

    def _preprocess_training_data(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
    ) -> Tuple:
        """RNNs require different preprocessing for variable length sequences."""

        idx = list(range(x.shape[0]))
        np.random.seed(self.random_state)
        np.random.shuffle(idx)

        x = get_padded_features(x)
        self.pad_size = x.shape[1]
        x_train_np, t_train_np, e_train_np = x[idx], t[idx], e[idx]

        x_train = torch.from_numpy(x_train_np.astype(float)).float().to(self.device)
        t_train = torch.from_numpy(t_train_np.astype(float)).float().to(self.device)
        e_train = torch.from_numpy(e_train_np.astype(int)).float().to(self.device)

        val_size = int(self.val_size * x_train.shape[0])

        x_val, t_val, e_val = x_train[-val_size:], t_train[-val_size:], e_train[-val_size:]

        x_train = x_train[:-val_size]
        t_train = t_train[:-val_size]
        e_train = e_train[:-val_size]

        return (x_train, t_train, e_train, x_val, t_val, e_val)

    def predict_emb(
        self,
        x: np.ndarray,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError(
                "The model has not been fitted yet. Please fit the "
                + "model using the `fit` method on some training data "
                + "before calling `predict_survival`."
            )
        x_in: torch.Tensor = self._preprocess_test_data(x)

        _, emb = self.model.forward_emb(x_in)

        return emb

    def predict_survival(
        self,
        x: np.ndarray,
        t: List,
        risk: int = 1,
        all_step: bool = False,
        bs: int = 100,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "The model has not been fitted yet. Please fit the "
                + "model using the `fit` method on some training data "
                + "before calling `predict_survival`."
            )
        lens = [len(x_) for x_ in x]

        x_in: Any = x
        if all_step:
            new_x = []
            for x_, l_ in zip(x, lens):
                new_x += [x_[: li + 1] for li in range(l_)]
            x_in = new_x

        # TODO: The below [t] is messy, need to investigate...
        t = self.discretize([t], self.split, self.split_time)[0][0]  # type: ignore

        x_in_tensor: torch.Tensor = self._preprocess_test_data(x_in)
        batches = int(len(x) / bs) + 1
        scores: dict = {t_: [] for t_ in t}
        for j in range(batches):
            xb = x_in_tensor[j * self.batch_size : (j + 1) * self.batch_size]
            _, f = self.model(xb)  # pylint: disable=not-callable
            for t_ in t:
                pred = torch.cumsum(f[int(risk) - 1], dim=1)[:, t_].squeeze().detach().cpu().numpy().tolist()

                if isinstance(pred, list):
                    scores[t_].extend(pred)
                else:  # pragma: no cover
                    scores[t_].append(pred)

        output = []
        for t_ in t:
            output.append(scores[t_])

        return 1 - np.asarray(output).T

    def predict_risk(self, x: np.ndarray, t: List, **kwargs: Any) -> np.ndarray:
        return 1 - self.predict_survival(x, t, **kwargs)

    def negative_log_likelihood(
        self,
        outcomes: torch.Tensor,
        cif: List[torch.Tensor],
        t: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log likelihood loss.

        This function is used to compute the survival loss.
        """
        loss: torch.Tensor = 0.0  # type: ignore
        censored_cif: torch.Tensor = 0.0  # type: ignore
        for k, ok in enumerate(outcomes):
            # Censored cif
            censored_cif += cif[k][e == 0][:, t[e == 0]]

            # Uncensored
            selection = e == (k + 1)
            loss += torch.sum(torch.log(ok[selection][:, t[selection]] + constants.EPS))

        # Censored loss
        loss += torch.sum(torch.log(nn.ReLU()(1 - censored_cif) + constants.EPS))
        return -loss / len(outcomes)

    def ranking_loss(
        self,
        cif: List[torch.Tensor],
        t: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize wrong ordering of probability.

        Equivalent to a C Index. This function is used to penalize wrong ordering in the survival prediction.
        """
        loss: torch.Tensor = 0.0  # type: ignore
        # Data ordered by time
        for k, cif_k in enumerate(cif):
            for ci, ti in zip(cif_k[e - 1 == k], t[e - 1 == k]):
                # For all events: all patients that didn't experience event before
                # must have a lower risk for that cause
                if torch.sum(t > ti) > 0:
                    # TODO: When data are sorted in time -> wan we make it even faster?
                    loss += torch.mean(torch.exp((cif_k[t > ti][:, ti] - ci[ti])) / self.sigma)

        return loss / len(cif)

    def longitudinal_loss(self, longitudinal_prediction: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Penalize error in the longitudinal predictions. This function is used to compute the error made by the RNN.

        NB: In the paper, they seem to use different losses for continuous and categorical,
        but this was not reflected in the code associated (therefore we compute MSE for all).

        NB: Original paper mentions possibility of different alphas for each risk,
        but takes same for all (for ranking loss).
        """
        length = (~torch.isnan(x[:, :, 0])).sum(dim=1) - 1

        # Create a grid of the column index
        index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(self.device)

        # Select all predictions until the last observed
        prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))

        # Select all observations that can be predicted
        observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
        observation_mask[:, 0] = False  # Remove first observation

        return torch.nn.MSELoss(reduction="mean")(longitudinal_prediction[prediction_mask], x[observation_mask])

    def total_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Invalid model for loss")

        longitudinal_prediction, outcomes = self.model(x.float())  # pylint: disable=not-callable
        if torch.isnan(longitudinal_prediction).sum() != 0:
            raise RuntimeError("NaNs detected in the longitudinal_prediction")

        t, e = t.long(), e.int()

        # Compute cumulative function from predicted outcomes
        cif = [torch.cumsum(ok, 1) for ok in outcomes]

        return (
            (1 - self.alpha - self.beta) * self.longitudinal_loss(longitudinal_prediction, x)
            + self.alpha * self.ranking_loss(cif, t, e)
            + self.beta * self.negative_log_likelihood(outcomes, cif, t, e)
        )


class DynamicDeepHitLayers(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        output_dim: int,
        layers_rnn: int,
        hidden_rnn: int,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
        risks: int = 1,
        output_type: str = "MLP",
        device: Any = constants.DEVICE,
    ) -> None:
        super(DynamicDeepHitLayers, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.risks = risks
        self.rnn_type = rnn_type
        self.device = device
        self.dropout = dropout

        # RNN model for longitudinal data
        self.embedding: nn.Module
        if self.rnn_type == "LSTM":
            self.embedding = nn.LSTM(input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True)
        elif self.rnn_type == "RNN":
            self.embedding = nn.RNN(
                input_dim,
                hidden_rnn,
                layers_rnn,
                bias=False,
                batch_first=True,
                nonlinearity="relu",
            )
        elif self.rnn_type == "GRU":
            self.embedding = nn.GRU(input_dim, hidden_rnn, layers_rnn, bias=False, batch_first=True)
        elif self.rnn_type == "Transformer":
            self.embedding = TransformerModel(input_dim, hidden_rnn, n_layers_hidden=layers_rnn, dropout=dropout)
        else:
            raise RuntimeError(f"Unknown rnn_type {rnn_type}")

        # Longitudinal network
        self.longitudinal = MLP(
            task_type="regression",
            n_units_in=hidden_rnn,
            n_units_out=input_dim,
            n_layers_hidden=layers_rnn,
            n_units_hidden=hidden_rnn,
            dropout=self.dropout,
            device=device,
        )

        # Attention mechanism
        self.attention: Union[MLP, TimeSeriesLayer]
        if output_type == "MLP":
            self.attention = MLP(
                task_type="regression",
                n_units_in=input_dim + hidden_rnn,
                n_units_out=1,
                dropout=self.dropout,
                n_layers_hidden=layers_rnn,
                n_units_hidden=hidden_rnn,
                device=device,
            )
        else:
            self.attention = TimeSeriesLayer(
                n_static_units_in=0,
                n_temporal_units_in=input_dim + hidden_rnn,
                n_temporal_window=seq_len,
                n_units_out=seq_len,
                n_temporal_units_hidden=hidden_rnn,
                n_temporal_layers_hidden=layers_rnn,
                mode=output_type,
                dropout=self.dropout,
                device=device,
            )
        self.attention_soft = nn.Softmax(1)  # On temporal dimension
        self.output_type = output_type

        # Cause specific network
        cause_specific = []
        for r in range(self.risks):  # pylint: disable=unused-variable
            cause_specific.append(
                MLP(
                    task_type="regression",
                    n_units_in=input_dim + hidden_rnn,
                    n_units_out=output_dim,
                    dropout=self.dropout,
                    n_layers_hidden=layers_rnn,
                    n_units_hidden=hidden_rnn,
                    device=device,
                )
            )
        self.cause_specific = nn.ModuleList(cause_specific)

        # Probability
        self.soft = nn.Softmax(dim=-1)  # On all observed output

        self.to(self.device)

    def forward_attention(self, x: torch.Tensor, inputmask: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Attention using last observation to predict weight of all previously observed
        # Extract last observation (the one used for predictions)
        last_observations = (~inputmask).sum(dim=1) - 1
        last_observations_idx = last_observations.unsqueeze(1).repeat(1, x.size(1))
        index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(self.device)

        last = index == last_observations_idx
        x_last = x[last]

        # Concatenate all previous with new to measure attention
        concatenation = torch.cat([hidden, x_last.unsqueeze(1).repeat(1, x.size(1), 1)], -1)

        # Compute attention and normalize
        if self.output_type == "MLP":
            attention = self.attention(concatenation).squeeze(-1)
        else:
            attention = self.attention(torch.zeros(len(concatenation), 0).to(self.device), concatenation).squeeze(-1)
        attention[index >= last_observations_idx] = -1e10  # Want soft max to be zero as values not observed
        attention[last_observations > 0] = self.attention_soft(
            attention[last_observations > 0]
        )  # Weight previous observation
        attention[last_observations == 0] = 0  # No context for only one observation

        # Risk networks
        # The original paper is not clear on how the last observation is
        # combined with the temporal sum, other code was concatenating them
        attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))
        hidden_attentive = torch.sum(attention * hidden, dim=1)
        return torch.cat([hidden_attentive, x_last], 1)

    def forward_emb(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward function that is called when data is passed through DynamicDeepHit."""
        # RNN representation - Nan values for not observed data
        x = x.clone()
        inputmask = torch.isnan(x[:, :, 0])
        x[torch.isnan(x)] = -1

        if torch.isnan(x).sum() != 0:  # pragma: no cover
            raise RuntimeError("NaNs detected in the input")

        if self.rnn_type in ["GRU", "LSTM", "RNN"]:
            hidden, _ = self.embedding(x)
        else:
            hidden = self.embedding(x)

        if torch.isnan(hidden).sum() != 0:  # pragma: no cover
            raise RuntimeError("NaNs detected in the embeddings")

        # Longitudinal modelling
        longitudinal_prediction = self.longitudinal(hidden)
        if torch.isnan(longitudinal_prediction).sum() != 0:  # pragma: no cover
            raise RuntimeError("NaNs detected in the longitudinal_prediction")

        hidden_attentive = self.forward_attention(x, inputmask, hidden)

        return longitudinal_prediction, hidden_attentive

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """The forward function that is called when data is passed through DynamicDeepHit."""
        # RNN representation - Nan values for not observed data
        x = x.to(self.device)

        longitudinal_prediction, hidden_attentive = self.forward_emb(x)

        outcomes = []
        for cs_nn in self.cause_specific:
            outcomes.append(cs_nn(hidden_attentive))

        # Soft max for probability distribution
        outcomes_t = torch.cat(outcomes, dim=1)
        outcomes_t = self.soft(outcomes_t)
        if torch.isnan(outcomes_t).sum() != 0:  # pragma: no cover
            raise RuntimeError("NaNs detected in the outcome")

        outcomes = [outcomes_t[:, i * self.output_dim : (i + 1) * self.output_dim] for i in range(self.risks)]
        return longitudinal_prediction, outcomes
