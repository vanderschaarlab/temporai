from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sklearn.datasets import load_diabetes, load_digits

from tempor.models.constants import ModelTaskType, Nonlin
from tempor.models.mlp import MLP, LinearLayer, MultiActivationHead, ResidualLayer


def test_network_config() -> None:
    net = MLP(
        task_type="regression",
        n_units_in=10,
        n_units_out=2,
        n_layers_hidden=2,
        n_units_hidden=20,
        batch_size=23,
        n_iter=34,
        lr=1e-2,
        dropout=0.5,
        batch_norm=True,
        nonlin="elu",
        patience=66,
        random_state=77,
    )

    assert len(net.model) == 3
    assert net.batch_size == 23
    assert net.n_iter == 34
    assert net.lr == 1e-2
    assert net.patience == 66
    assert net.random_state == 77


@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("n_iter", [10, 50, 100])
@pytest.mark.parametrize("dropout", [0, 0.5, 0.2])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("lr", [1e-3, 3e-4])
@pytest.mark.parametrize("residual", [True, False])
def test_basic_network(
    task_type: ModelTaskType,
    nonlin: Nonlin,
    n_iter: int,
    dropout: float,
    batch_norm: bool,
    lr: float,
    residual: bool,
) -> None:
    net = MLP(
        task_type=task_type,
        n_units_in=10,
        n_units_out=2,
        n_iter=n_iter,
        dropout=dropout,
        nonlin=nonlin,
        batch_norm=batch_norm,
        n_layers_hidden=2,
        lr=lr,
        residual=residual,
    )

    assert net.n_iter == n_iter
    assert net.task_type == task_type
    assert net.lr == lr


@pytest.mark.parametrize("layer", [LinearLayer, ResidualLayer])
def test_custom_layers(layer: torch.nn.Module) -> None:
    X, _ = load_digits(return_X_y=True)
    Xt = torch.from_numpy(X)

    mod = layer(Xt.shape[1], 10).cpu()
    assert mod(Xt).shape[0] == Xt.shape[0]
    assert mod(Xt).shape[1] >= 10


@pytest.mark.parametrize(
    "activations",
    [
        [(torch.nn.ReLU(), 10), (torch.nn.Softmax(dim=-1), 30), (torch.nn.Tanh(), 24)],
        [(torch.nn.ReLU(), 64)],
        [(torch.nn.ReLU(), 1) for i in range(64)],
    ],
)
def test_multiactivation_heads(activations: list) -> None:
    X, _ = load_digits(return_X_y=True)
    Xt = torch.from_numpy(X)

    mod = MultiActivationHead(activations=activations)
    assert mod(Xt).shape == Xt.shape


@pytest.mark.parametrize(
    "activations",
    [
        [(torch.nn.ReLU(), 10), (torch.nn.Softmax(dim=-1), 30), (torch.nn.Tanh(), 2)],
        [(torch.nn.ReLU(), 1)],
        [(torch.nn.ReLU(), 1) for i in range(65)],
    ],
)
def test_multiactivation_heads_failure(activations: list) -> None:
    X, _ = load_digits(return_X_y=True)
    Xt = torch.from_numpy(X)

    with pytest.raises(RuntimeError):
        MultiActivationHead(activations=activations)(Xt)


@pytest.mark.parametrize("residual", [True, False])
def test_mlp_classification(residual: bool) -> None:
    X, y = load_digits(return_X_y=True)
    if TYPE_CHECKING:  # pragma: no cover
        assert isinstance(X, np.ndarray)  # nosec B101
        assert isinstance(y, np.ndarray)  # nosec B101

    model = MLP(
        task_type="classification",
        n_units_in=X.shape[1],
        n_units_out=len(np.unique(y)),
        residual=residual,
        n_iter=10,
    )

    model.fit(X, y)

    assert model.predict(X).shape == y.shape
    assert model.predict_proba(X).shape == (len(y), 10)


@pytest.mark.parametrize("residual", [True, False])
def test_mlp_regression(residual: bool) -> None:
    X, y = load_diabetes(return_X_y=True)
    if TYPE_CHECKING:  # pragma: no cover
        assert isinstance(X, np.ndarray)  # nosec B101
        assert isinstance(y, np.ndarray)  # nosec B101

    model = MLP(
        task_type="regression",
        n_units_in=X.shape[1],
        n_units_out=1,
        residual=residual,
        n_iter=10,
    )

    model.fit(X, y)

    assert model.predict(X).shape == y.shape
    with pytest.raises(ValueError):
        model.predict_proba(X)


def test_mlp_input_validation_fails():
    with pytest.raises(ValueError, match=".*>=.*0.*"):
        MLP(task_type="regression", n_units_in=-1, n_units_out=3)
    with pytest.raises(ValueError, match=".*>=.*0.*"):
        MLP(task_type="regression", n_units_in=3, n_units_out=-1)


def test_mlp_hidden_0():
    model = MLP(task_type="regression", n_units_in=3, n_units_out=3, n_layers_hidden=0)
    assert len(model.model) == 1


def test_nonlin_out_mismatch():
    with pytest.raises(RuntimeError, match=".*mismatch.*"):
        MLP(task_type="regression", n_units_in=3, n_units_out=3, nonlin_out=[("tanh", 1), ("softmax", 3)])


def test_mlp_loss_provided():
    mock_loss = Mock()
    model = MLP(task_type="regression", n_units_in=3, n_units_out=3, n_layers_hidden=0, loss=mock_loss)
    assert model.loss == mock_loss


def test_residual_layer_x_last_dim_0():
    rl = ResidualLayer(n_units_in=10, n_units_out=10)
    out = rl.forward(torch.ones(size=(3, 3, 0)))
    assert list(out.shape) == [3, 3, 10]


def test_mlp_score():
    model = MLP(task_type="regression", n_units_in=3, n_units_out=3, n_units_hidden=3, device=torch.device("cpu"))
    out = model.score(X=np.ones(shape=(3, 3)), y=np.ones((3,)))
    assert isinstance(out, float)

    model = MLP(task_type="classification", n_units_in=3, n_units_out=3, n_units_hidden=3, device=torch.device("cpu"))
    out = model.score(X=np.ones(shape=(3, 3)), y=np.ones((3,)))
    assert isinstance(out, float)


def test_mlp_clipping_0_case():
    X, y = load_digits(return_X_y=True)

    model = MLP(
        task_type="classification",
        n_units_in=X.shape[1],
        n_units_out=len(np.unique(y)),
        n_iter=10,
        clipping_value=0,
    )

    model.fit(X, y)  # type: ignore


def test_mlp_print_iter_no_early_stopping_case():
    X, y = load_digits(return_X_y=True)

    model = MLP(
        task_type="classification",
        n_units_in=X.shape[1],
        n_units_out=len(np.unique(y)),
        n_iter=10,
        n_iter_print=3,
        early_stopping=False,
    )

    model.fit(X, y)  # type: ignore


def test_mlp_early_stop():
    X, y = load_digits(return_X_y=True)

    model = MLP(
        task_type="classification",
        n_units_in=X.shape[1],
        n_units_out=len(np.unique(y)),
        early_stopping=True,
        n_iter=100,
        n_iter_min=0,
        patience=1,
    )

    model.fit(X, y)  # type: ignore


def test_mlp_len():
    X, y = load_digits(return_X_y=True)

    model = MLP(
        task_type="classification",
        n_units_in=X.shape[1],
        n_units_out=len(np.unique(y)),
        early_stopping=True,
        n_iter=10,
    )

    assert len(model) == len(model.model)


def test_linear_layer_nonlin_none():
    ll = LinearLayer(n_units_in=10, n_units_out=10, nonlin=None)
    assert len(ll.model) == 1
