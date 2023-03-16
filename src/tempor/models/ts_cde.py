from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pydantic
import torch
import torchcde
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, sampler

from tempor.log import logger as log
from tempor.models.constants import DEVICE, ModelTaskType, Nonlin
from tempor.models.mlp import MLP
from tempor.models.samplers import ImbalancedDatasetSampler
from tempor.models.utils import enable_reproducibility


class CDEFunc(torch.nn.Module):
    """CDEFunc computes f_\\theta for the CDE model : z_t = z_0 + \\int_0^t f_\\theta(z_s) dX_s

    Args:
        input_din
        n_units_hidden
    """

    def __init__(
        self,
        n_units_in: int,
        n_units_hidden: int,
        n_layers_hidden: int = 1,
        nonlin: Nonlin = "relu",
        dropout: float = 0,
        device: Any = DEVICE,
    ):
        super(CDEFunc, self).__init__()
        self.n_units_in = n_units_in
        self.n_units_hidden = n_units_hidden

        n_units_out = n_units_in * n_units_hidden
        self.model = MLP(
            task_type="regression",
            n_units_in=n_units_hidden,
            n_units_out=n_units_out,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
            nonlin_out=[("tanh", n_units_out)],
        )

    def forward(self, t, z):
        z = self.model(z)

        z = z.view(*z.shape[:-1], self.n_units_hidden, self.n_units_in)
        return z


class NeuralCDE(torch.nn.Module):
    """The model that computes the integral in : z_t = z_0 + \\int_0^t f_\\theta(z_s) dX_s

    Parameters
    ----------
    task_type: str
        classification or regression
    n_static_units_in: int
        Number of features in the static tensor
    n_temporal_units_in: int
        Number of features in the temporal tensor
    output_shape (List[int]):
        Shape of the output tensor.
    n_layers_hidden: int
        Number of hidden layers
    n_units_hidden: int
        Number of hidden units in each layer
    nonlin: string, default 'relu'
        Nonlinearity to use in NN. Can be 'elu', 'relu', 'selu', 'tanh' or 'leaky_relu'.
    nonlin_out (Optional[List[Tuple[Nonlin, int]]], optional):
        List of activations for the output. Example ``[("tanh", 1), ("softmax", 3)]`` - means the output layer
        will apply ``"tanh"`` for the first unit, and ``"softmax"`` for the following 3 units in the output.
        Defaults to `None`.
    # CDE specific
    atol: float
        absolute tolerance for solution
    rtol: float
        relative tolerance for solution
    interpolation: str
        cubic or linear
    # training
    lr: float
        learning rate for optimizer.
    weight_decay: float
        l2 (ridge) penalty for the weights.
    n_iter: int
        Maximum number of iterations.
    batch_size: int
        Batch size
    n_iter_print: int
        Number of iterations after which to print updates and check the validation loss.
    random_state: int
        random_state used
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    dropout: float
        Dropout value. If 0, the dropout is not used.
    clipping_value: int, default 1
        Gradients clipping value
    early_stopping: bool
        Enable/disable early stopping
    dataloader_sampler (Optional[sampler.Sampler], optional):
        Custom data sampler for training. Defaults to None.
    """

    def __init__(
        self,
        task_type: ModelTaskType,
        n_static_units_in: int,
        n_temporal_units_in: int,
        output_shape: List[int],
        n_units_hidden: int = 100,
        n_layers_hidden: int = 1,
        nonlin: Nonlin = "relu",
        nonlin_out: Optional[List[Tuple[Nonlin, int]]] = None,
        dropout: float = 0,
        # CDE specific
        atol: float = 1e-2,
        rtol: float = 1e-2,
        interpolation: str = "cubic",
        # training
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        opt_betas: tuple = (0.9, 0.999),
        n_iter: int = 1000,
        batch_size: int = 500,
        n_iter_print: int = 100,
        random_state: int = 0,
        patience: int = 10,
        n_iter_min: int = 100,
        clipping_value: int = 1,
        early_stopping: bool = True,
        train_ratio: float = 0.8,
        device: Any = DEVICE,
        dataloader_sampler: Optional[sampler.Sampler] = None,
    ):
        super(NeuralCDE, self).__init__()

        enable_reproducibility(random_state)
        if len(output_shape) == 0:
            raise ValueError("Invalid output shape")

        self.task_type = task_type

        self.func = CDEFunc(
            n_temporal_units_in + 1,  # we add the observation times
            n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            nonlin=nonlin,
            dropout=dropout,
            device=device,
        )
        self.initial_temporal = MLP(
            task_type="regression",
            n_units_in=n_temporal_units_in + 1,  # we add the observation times
            n_units_out=n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
        )

        self.output_shape = output_shape
        self.n_units_out = int(np.prod(self.output_shape))

        output_input_size = n_units_hidden
        self.initial_static: Optional[MLP] = None
        if n_static_units_in > 0:
            self.initial_static = MLP(
                task_type="regression",
                n_units_in=n_static_units_in,
                n_units_out=n_units_hidden,
                n_layers_hidden=n_layers_hidden,
                n_units_hidden=n_units_hidden,
                dropout=dropout,
                nonlin=nonlin,
                device=device,
            )
            output_input_size += n_units_hidden

        self.output = MLP(
            task_type=task_type,
            n_units_in=output_input_size,
            n_units_out=self.n_units_out,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
            nonlin_out=nonlin_out,
        )

        # CDE specific
        self.atol = atol
        self.rtol = rtol
        self.interpolation = interpolation

        # training
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.patience = patience
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping
        self.device = device
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.dataloader_sampler = dataloader_sampler

        if task_type == "classification":
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )  # optimize all rnn parameters

    def forward(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
    ) -> torch.Tensor:
        # sanity
        if torch.isnan(static_data).sum() != 0:
            raise ValueError("NaNs detected in the static data")
        if torch.isnan(temporal_data).sum() != 0:
            raise ValueError("NaNs detected in the temporal data")
        if torch.isnan(observation_times).sum() != 0:
            raise ValueError("NaNs detected in the temporal horizons")

        # Include the observation times as a channel in the dataset
        temporal_data_ext = torch.cat([temporal_data, observation_times.unsqueeze(-1)], dim=-1)

        # Convert the dataset into a continuous path.
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(temporal_data_ext)

        # Interpolate the input
        if self.interpolation == "linear":
            spline = torchcde.LinearInterpolation(coeffs)
        elif self.interpolation == "cubic":
            spline = torchcde.CubicSpline(coeffs)
        else:
            raise RuntimeError(f"Invalid interpolation {self.interpolation}")

        #  Initial hidden state should be a function of the first observation.
        X0 = spline.evaluate(spline.interval[0])
        z0 = self.initial_temporal(X0)

        # Solve the CDE
        z_T = torchcde.cdeint(X=spline, func=self.func, z0=z0, t=spline.interval, atol=self.atol, rtol=self.rtol)
        z_T = z_T[:, 1]

        # Compute static embedding
        if static_data is not None and self.initial_static is not None:
            static_emb = self.initial_static(static_data)
            z_T = torch.cat([z_T, static_emb], dim=-1)

        out = self.output(z_T)
        return out.reshape(-1, *self.output_shape)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        static_data: Union[List, np.ndarray, torch.Tensor],
        temporal_data: Union[List, np.ndarray, torch.Tensor],
        observation_times: Union[List, np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            (
                static_data_t,
                temporal_data_t,
                observation_times_t,
                _,
                window_batches,
            ) = self._prepare_input(static_data, temporal_data, observation_times)

            yt = torch.zeros(len(temporal_data), *self.output_shape).to(self.device)
            for widx in range(len(temporal_data_t)):
                window_size = len(observation_times_t[widx][0])
                local_yt = self(
                    static_data_t[widx],
                    temporal_data_t[widx],
                    observation_times_t[widx],
                )
                yt[window_batches[window_size]] = local_yt

            if self.task_type == "classification":
                return np.argmax(yt.cpu().numpy(), -1)
            else:
                return yt.cpu().numpy()

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_proba(
        self,
        static_data: Union[List, np.ndarray, torch.Tensor],
        temporal_data: Union[List, np.ndarray, torch.Tensor],
        observation_times: Union[List, np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        self.eval()
        if self.task_type != "classification":
            raise RuntimeError("Task valid only for classification")
        with torch.no_grad():
            (
                static_data_t,
                temporal_data_t,
                observation_times_t,
                _,
                window_batches,
            ) = self._prepare_input(static_data, temporal_data, observation_times)

            yt = torch.zeros(len(temporal_data), *self.output_shape).to(self.device)
            for widx in range(len(temporal_data_t)):
                window_size = len(observation_times_t[widx][0])
                local_yt = self(
                    static_data_t[widx],
                    temporal_data_t[widx],
                    observation_times_t[widx],
                )
                yt[window_batches[window_size]] = local_yt

            return yt.cpu().numpy()

    def score(
        self,
        static_data: Union[List, np.ndarray, torch.Tensor],
        temporal_data: Union[List, np.ndarray, torch.Tensor],
        observation_times: Union[List, np.ndarray, torch.Tensor],
        outcome: Union[List, np.ndarray],
    ) -> float:
        y_pred = self.predict(static_data, temporal_data, observation_times)
        outcome = np.asarray(outcome)
        if self.task_type == "classification":
            return np.mean(y_pred.astype(int) == outcome.astype(int))
        else:
            return np.mean(np.inner(outcome - y_pred, outcome - y_pred) / 2.0)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static_data: Union[List, np.ndarray, torch.Tensor],
        temporal_data: Union[List, np.ndarray, torch.Tensor],
        observation_times: Union[List, np.ndarray, torch.Tensor],
        outcome: Union[List, np.ndarray, torch.Tensor],
    ) -> Any:
        (
            static_data_t,
            temporal_data_t,
            observation_times_t,
            outcome_t,
            _,
        ) = self._prepare_input(static_data, temporal_data, observation_times, outcome)

        return self._train(static_data_t, temporal_data_t, observation_times_t, outcome_t)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _train(
        self,
        static_data: List[torch.Tensor],
        temporal_data: List[torch.Tensor],
        observation_times: List[torch.Tensor],
        outcome: List[torch.Tensor],
    ) -> Any:
        patience = 0
        prev_error = np.inf

        train_dataloaders = []
        test_dataloaders = []
        for widx in range(len(temporal_data)):
            train_dl, test_dl = self.dataloader(
                static_data[widx],
                temporal_data[widx],
                observation_times[widx],
                outcome[widx],
            )
            train_dataloaders.append(train_dl)
            test_dataloaders.append(test_dl)

        # training and testing
        for it in range(self.n_iter):
            train_loss = self._train_epoch(train_dataloaders)
            if it % self.n_iter_print == 0:
                val_loss = self._test_epoch(test_dataloaders)
                log.info(f"Epoch:{it}| train loss: {train_loss}, validation loss: {val_loss}")
                if val_loss < prev_error:
                    patience = 0
                    prev_error = val_loss
                else:
                    patience += 1
                if patience > self.patience:
                    break

        return self

    def _train_epoch(self, loaders: List[DataLoader]) -> float:
        self.train()

        losses = []
        for loader in loaders:
            for step, (static_mb, temporal_mb, horizons_mb, y_mb) in enumerate(  # pylint: disable=unused-variable
                loader
            ):
                self.optimizer.zero_grad()  # clear gradients for this training step

                pred = self(static_mb, temporal_mb, horizons_mb)  # rnn output

                loss = self.loss(pred.squeeze(), y_mb.squeeze())

                loss.backward()  # backpropagation, compute gradients
                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)  # pyright: ignore
                self.optimizer.step()  # apply gradients

                losses.append(loss.detach().cpu())

        return float(np.mean(losses))

    def _test_epoch(self, loaders: List[DataLoader]) -> float:
        self.eval()

        losses = []
        for loader in loaders:
            for step, (static_mb, temporal_mb, horizons_mb, y_mb) in enumerate(  # pylint: disable=unused-variable
                loader
            ):
                pred = self(static_mb, temporal_mb, horizons_mb)  # rnn output
                loss = self.loss(pred.squeeze(), y_mb.squeeze())

                losses.append(loss.detach().cpu())

        return float(np.mean(losses))

    def dataloader(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        outcome: torch.Tensor,
    ) -> Tuple[DataLoader, DataLoader]:
        stratify = None
        _, out_counts = torch.unique(outcome, return_counts=True)
        if out_counts.min() > 1:
            stratify = outcome.cpu()

        split: Tuple[torch.Tensor, ...] = train_test_split(  # type: ignore
            static_data.cpu(),
            temporal_data.cpu(),
            observation_times.cpu(),
            outcome.cpu(),
            train_size=self.train_ratio,
            random_state=self.random_state,
            stratify=stratify,
        )
        (
            static_data_train,
            static_data_test,
            temporal_data_train,
            temporal_data_test,
            observation_times_train,
            observation_times_test,
            outcome_train,
            outcome_test,
        ) = split
        train_dataset = TensorDataset(
            static_data_train.to(self.device),
            temporal_data_train.to(self.device),
            observation_times_train.to(self.device),
            outcome_train.to(self.device),
        )
        test_dataset = TensorDataset(
            static_data_test.to(self.device),
            temporal_data_test.to(self.device),
            observation_times_test.to(self.device),
            outcome_test.to(self.device),
        )

        sampler_ = self.dataloader_sampler
        if sampler_ is None and self.task_type == "classification":
            sampler_ = ImbalancedDatasetSampler(outcome_train.squeeze().cpu().numpy().tolist())

        return (
            DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler_,
                pin_memory=False,
            ),
            DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                pin_memory=False,
            ),
        )

    def _check_tensor(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _prepare_input(
        self,
        static_data: Union[List, np.ndarray, torch.Tensor],
        temporal_data: Union[List, np.ndarray, torch.Tensor],
        observation_times: Union[List, np.ndarray, torch.Tensor],
        outcome: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
    ) -> Tuple:
        static_data = np.asarray(static_data)
        temporal_data = np.asarray(temporal_data)
        observation_times = np.asarray(observation_times)
        if outcome is not None:
            outcome = np.asarray(outcome)

        window_batches: Dict[int, List[int]] = {}
        for idx, item in enumerate(observation_times):
            window_len = len(item)
            if window_len not in window_batches:
                window_batches[window_len] = []
            window_batches[window_len].append(idx)

        static_data_mb = []
        temporal_data_mb = []
        observation_times_mb = []
        outcome_mb = []

        for widx in window_batches:
            indices = window_batches[widx]

            static_data_t = self._check_tensor(static_data[indices]).float()

            local_temporal_data = np.array(temporal_data[indices].tolist()).astype(float)
            temporal_data_t = self._check_tensor(local_temporal_data).float()
            local_observation_times = np.array(observation_times[indices].tolist()).astype(float)
            observation_times_t = self._check_tensor(local_observation_times).float()

            static_data_mb.append(static_data_t)
            temporal_data_mb.append(temporal_data_t)
            observation_times_mb.append(observation_times_t)

            if outcome is not None:
                outcome_t = self._check_tensor(outcome[indices]).float()

                if self.task_type == "classification":
                    outcome_t = outcome_t.long()
                outcome_mb.append(outcome_t)

        return (
            static_data_mb,
            temporal_data_mb,
            observation_times_mb,
            outcome_mb,
            window_batches,
        )
