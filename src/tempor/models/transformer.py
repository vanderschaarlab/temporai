"""Model implementations for Transformers."""

from typing import Any

import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from .constants import DEVICE


class Permute(nn.Module):
    def __init__(self, *dims: Any) -> None:
        """Permute dimensions of a tensor with `torch.Tensor.permute`.

        Args:
            *dims (Any):
                Dimensions to permute.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.permute(self.dims)


class Transpose(nn.Module):
    def __init__(self, *dims: Any, contiguous: bool = False) -> None:
        """Transpose dimensions of a tensor with `torch.Tensor.transpose`.

        Args:
            *dims (Any): Dimensions to transpose.
            contiguous (bool, optional): Whether to call `.contiguous()` on the output. Defaults to `False`.
        """
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_hidden: int = 64,
        n_head: int = 1,
        d_ffn: int = 128,
        dropout: float = 0.1,
        activation: str = "relu",
        n_layers_hidden: int = 1,
        device: Any = DEVICE,
    ) -> None:
        """Transformer model.

        Args:
            n_units_in (int):
                The number of features (a.k.a. variables, dimensions, channels) in the time series dataset.
            n_units_hidden (int, optional):
                Total dimension of the model. Defaults to ``64``.
            n_head (int, optional):
                Parallel attention heads. Defaults to ``1``.
            d_ffn (int, optional):
                The dimension of the feedforward network model. Defaults to ``128``.
            dropout (float, optional):
                Dropout value passed to `~torch.nn.modules.transformer.TransformerEncoderLayer` s. Defaults to ``0.1``.
            activation (str, optional):
                The activation function of intermediate layer, ``"relu"`` or ``"gelu"``. Defaults to ``"relu"``.
            n_layers_hidden (int, optional):
                The number of sub-encoder-layers in the encoder. Defaults to ``1``.
            device (Any, optional):
                PyTorch device. Defaults to `~tempor.models.constants.DEVICE`.
        """

        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            n_units_hidden,
            n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
        )
        encoder_norm = nn.LayerNorm(n_units_hidden)
        self.transformer_encoder = TransformerEncoder(  # type: ignore [no-untyped-call]
            encoder_layer,
            n_layers_hidden,
            norm=encoder_norm,
        )

        self.model = nn.Sequential(
            Permute(1, 0, 2),  # bs x seq_len x nvars -> seq_len x bs x nvars
            nn.Linear(n_units_in, n_units_hidden),  # seq_len x bs x nvars -> seq_len x bs x n_units_hidden
            nn.ReLU(),
            self.transformer_encoder,
            Transpose(1, 0),  # seq_len x bs x n_units_hidden -> bs x seq_len x n_units_hidden
            nn.ReLU(),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)  # pylint: disable=not-callable
