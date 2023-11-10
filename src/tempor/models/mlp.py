"""Model implementations related to MLP / fully connected neural networks."""

from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import pydantic
import torch
import torch.utils.data
from torch import nn

from tempor.core import pydantic_utils
from tempor.log import logger
from tempor.models import constants, utils

from .constants import Nonlin
from .utils import GumbelSoftmax, get_nonlin


class LinearLayer(nn.Module):
    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
        nonlin: Optional[Nonlin] = "relu",
        device: Any = constants.DEVICE,
    ) -> None:
        """Linear layer with optional dropout, batch norm, and nonlinearity.

        Args:
            n_units_in (int): Number of input units.
            n_units_out (int): Number of output units.
            dropout (float, optional): Dropout. Defaults to ``0.0``.
            batch_norm (bool, optional): Batch normalization. Defaults to `False`.
            nonlin (Optional[Nonlin], optional): Nonlinearity (activation function). Defaults to ``"relu"``.
            device (Any, optional): Device. Defaults to ``constants.DEVICE``.
        """
        super(LinearLayer, self).__init__()

        self.device = device
        layers: List[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(n_units_in, n_units_out))

        if batch_norm:
            layers.append(nn.BatchNorm1d(n_units_out))

        if nonlin is not None:
            layers.append(get_nonlin(nonlin))

        self.model = nn.Sequential(*layers).to(self.device)

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(X.float()).to(self.device)  # pylint: disable=not-callable


class ResidualLayer(LinearLayer):
    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        dropout: float = 0.0,
        batch_norm: bool = False,
        nonlin: Optional[Nonlin] = "relu",
        device: Any = constants.DEVICE,
    ) -> None:
        """Residual layer.

        Args:
            n_units_in (int): Number of input units.
            n_units_out (int): Number of output units.
            dropout (float, optional): Dropout. Defaults to ``0.0``.
            batch_norm (bool, optional): Batch normalization. Defaults to `False`.
            nonlin (Optional[Nonlin], optional): Nonlinearity (activation function). Defaults to ``"relu"``.
            device (Any, optional): Device. Defaults to `constants.DEVICE`.
        """
        super(ResidualLayer, self).__init__(
            n_units_in,
            n_units_out,
            dropout=dropout,
            batch_norm=batch_norm,
            nonlin=nonlin,
            device=device,
        )
        self.device = device
        self.n_units_out = n_units_out

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if X.shape[-1] == 0:
            return torch.zeros((*X.shape[:-1], self.n_units_out)).to(self.device)

        out = self.model(X.float())  # pylint: disable=not-callable
        return torch.cat([out, X], dim=-1).to(self.device)


class MultiActivationHead(nn.Module):
    def __init__(
        self,
        activations: List[Tuple[nn.Module, int]],
        device: Any = constants.DEVICE,
    ) -> None:
        """Final layer with multiple activations. Useful for tabular data. The different activations are applied to
        different sub-lengths of the ``X`` tensor in the forward pass, on its final dimension. Hence the ``X`` tensor
        must have a shape of ``(..., sum(activation_lengths))``.

        Args:
            activations (List[Tuple[nn.Module, int]]):
                List of tuples of activations and their lengths, ``[(activation, activation_length), ...]``.
            device (Any, optional):
                Device. Defaults to `constants.DEVICE`.
        """
        super(MultiActivationHead, self).__init__()
        self.activations = []
        self.activation_lengths = []
        self.device = device

        for activation, length in activations:
            self.activations.append(activation)
            self.activation_lengths.append(length)

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if X.shape[-1] != np.sum(self.activation_lengths):
            raise RuntimeError(
                f"Shape mismatch for the activations: expected {np.sum(self.activation_lengths)}. Got shape {X.shape}."
            )

        split = 0
        out = torch.zeros(X.shape).to(self.device)

        for activation, step in zip(self.activations, self.activation_lengths):
            out[..., split : split + step] = activation(X[..., split : split + step])

            split += step

        return out


class MLP(nn.Module):
    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        task_type: constants.ModelTaskType,
        n_units_in: int,
        n_units_out: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: Nonlin = "relu",
        nonlin_out: Optional[List[Tuple[Nonlin, int]]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        opt_betas: Tuple[float, float] = (0.9, 0.999),
        n_iter: int = 1000,
        batch_size: int = 500,
        n_iter_print: int = 100,
        random_state: int = 0,
        patience: int = 10,
        n_iter_min: int = 100,
        dropout: float = 0.1,
        clipping_value: int = 1,
        batch_norm: bool = False,
        early_stopping: bool = True,
        residual: bool = False,
        loss: Optional[Callable] = None,
        device: Any = constants.DEVICE,
    ) -> None:
        """Fully connected or residual neural nets for classification and regression.

        Args:
            task_type (constants.ModelTaskType):
                The type of the problem. Available options: :obj:`~tempor.models.constants.ModelTaskType`.
            n_units_in (int):
                Number of features.
            n_units_out (int):
                Number of outputs.
            n_layers_hidden (int, optional):
                Number of hidden layers. Defaults to ``1``.
            n_units_hidden (int, optional):
                Number of hidden units in each layer. Defaults to ``100``.
            nonlin (Nonlin, optional):
                Nonlinearity to use in NN. Available options: :obj:`~tempor.models.constants.Nonlin`.
                Defaults to ``"relu"``.
            nonlin_out (Optional[List[Tuple[Nonlin, int]]], optional):
                List of activations for the output. Example ``[("tanh", 1), ("softmax", 3)]`` - means the output layer
                will apply ``"tanh"`` for the first unit, and ``"softmax"`` for the following 3 units in the output.
                Defaults to `None`.
            lr (float, optional):
                Learning rate. Defaults to ``1e-3``.
            weight_decay (float, optional):
                l2 (ridge) penalty for the weights. Defaults to ``1e-3``.
            opt_betas (Tuple[float, float], optional):
                Optimizer betas. Defaults to ``(0.9, 0.999)``.
            n_iter (int, optional):
                Maximum number of iterations. Defaults to ``1000``.
            batch_size (int, optional):
                Batch size. Defaults to ``500``.
            n_iter_print (int, optional):
                Number of iterations after which to print updates and check the validation loss. Defaults to ``100``.
            random_state (int, optional):
                Random seed. Defaults to ``0``.
            patience (int, optional):
                Number of iterations to wait before early stopping after decrease in validation loss.
                Defaults to ``10``.
            n_iter_min (int, optional):
                Minimum number of iterations to go through before starting early stopping. Defaults to ``100``.
            dropout (float, optional):
                Dropout value. If ``0``, the dropout is not used. Defaults to ``0.1``.
            clipping_value (int, optional):
                Gradients clipping value. Defaults to ``1``.
            batch_norm (bool, optional):
                Enable/disable batch normalization. Defaults to `False`.
            early_stopping (bool, optional):
                Enable/disable early stopping. Defaults to `True`.
            residual (bool, optional):
                Add residuals. Defaults to `False`.
            loss (Optional[Callable], optional):
                Optional custom loss function. If `None`, the loss is `torch.nn.CrossEntropyLoss` for classification
                tasks, or `torch.nn.MSELoss` for regression. Defaults to `None`.
            device (Any, optional):
                PyTorch device to use. Defaults to `~tempor.models.constants.DEVICE`.
        """
        super(MLP, self).__init__()

        if n_units_in < 0:
            raise ValueError("n_units_in must be >= 0")
        if n_units_out < 0:
            raise ValueError("n_units_out must be >= 0")

        utils.enable_reproducibility(random_state)
        self.device = device
        self.task_type = task_type
        self.random_state = random_state

        block: Type[LinearLayer]
        if residual:
            block = ResidualLayer
        else:
            block = LinearLayer

        # network
        layers: List[nn.Module] = []

        if n_layers_hidden > 0:
            layers.append(
                block(
                    n_units_in,
                    n_units_hidden,
                    batch_norm=batch_norm,
                    nonlin=nonlin,
                    device=device,
                )
            )
            n_units_hidden += int(residual) * n_units_in

            # add required number of layers
            for i in range(n_layers_hidden - 1):  # pylint: disable=unused-variable
                layers.append(
                    block(
                        n_units_hidden,
                        n_units_hidden,
                        batch_norm=batch_norm,
                        nonlin=nonlin,
                        dropout=dropout,
                        device=device,
                    )
                )
                n_units_hidden += int(residual) * n_units_hidden

            # add final layers
            layers.append(nn.Linear(n_units_hidden, n_units_out, device=device))
        else:
            layers = [nn.Linear(n_units_in, n_units_out, device=device)]

        if nonlin_out is not None:
            total_nonlin_len = 0
            activations = []
            for nonlin, nonlin_len in nonlin_out:
                total_nonlin_len += nonlin_len
                activations.append((get_nonlin(nonlin), nonlin_len))

            if total_nonlin_len != n_units_out:
                raise RuntimeError(
                    f"Shape mismatch for the output layer. Expected length {n_units_out}, but got {nonlin_out} "
                    f"with length {total_nonlin_len}"
                )
            layers.append(MultiActivationHead(activations, device=device))
        elif self.task_type == "classification":
            layers.append(MultiActivationHead([(GumbelSoftmax(), n_units_out)], device=device))

        self.model = nn.Sequential(*layers).to(self.device)

        # optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt_betas = opt_betas
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.opt_betas,
        )

        # training
        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.batch_size = batch_size
        self.patience = patience
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping
        if loss is not None:
            self.loss = loss
        else:
            if task_type == "classification":
                self.loss = nn.CrossEntropyLoss()
            else:
                self.loss = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        """Fit (train) the model."""
        Xt = self._check_tensor(X)
        yt = self._check_tensor(y)

        self._train(Xt, yt)

        return self

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for classification tasks.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.task_type != "classification":
            raise ValueError(f"Invalid task type for predict_proba {self.task_type}")

        with torch.no_grad():
            Xt = self._check_tensor(X)

            yt = self.forward(Xt)

            return yt.cpu().numpy().squeeze()

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        with torch.no_grad():
            Xt = self._check_tensor(X)

            yt = self.forward(Xt)

            if self.task_type == "classification":
                return np.argmax(yt.cpu().numpy().squeeze(), -1).squeeze()
            else:
                return yt.cpu().numpy().squeeze()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the default score of the model. See source code."""
        y_pred = self.predict(X)
        if self.task_type == "classification":
            return np.mean(y_pred == y)
        else:
            return np.mean(np.inner(y - y_pred, y - y_pred) / 2.0)

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(X.float())  # pylint: disable=not-callable

    def _train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        train_loss = []

        for batch_ndx, sample in enumerate(loader):  # pylint: disable=unused-variable
            self.optimizer.zero_grad()

            X_next, y_next = sample
            if len(X_next) < 2:  # pragma: no cover
                continue

            preds = self.forward(X_next).squeeze()

            batch_loss = self.loss(preds.squeeze(), y_next.squeeze())

            batch_loss.backward()

            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(  # type: ignore [attr-defined] # pyright: ignore
                    self.parameters(),
                    self.clipping_value,
                )

            self.optimizer.step()

            train_loss.append(batch_loss.detach())

        return torch.mean(torch.Tensor(train_loss)).item()

    def _train(self, X: torch.Tensor, y: torch.Tensor) -> "MLP":
        X = self._check_tensor(X).float()
        y = self._check_tensor(y).squeeze().float()
        if self.task_type == "classification":
            y = y.long()

        # Load Dataset
        dataset = torch.utils.data.TensorDataset(X, y)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=False)

        # Setup the network and optimizer

        val_loss_best = np.inf
        patience = 0

        # do training
        for i in range(self.n_iter):
            train_loss = self._train_epoch(loader)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():
                    X_val, y_val = test_dataset.dataset.tensors  # type: ignore

                    preds = self.forward(X_val).squeeze()
                    val_loss = self.loss(preds.squeeze(), y_val.squeeze())

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.patience and i > self.n_iter_min:
                            break

                    if i % self.n_iter_print == 0:
                        logger.debug(f"Epoch: {i}, loss: {val_loss}, train_loss: {train_loss}")

        return self

    def _check_tensor(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def __len__(self) -> int:
        """Return the number of layers in the model ``len(self.model)``."""
        return len(self.model)
