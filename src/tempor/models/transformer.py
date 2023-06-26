from typing import Any

import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from .constants import DEVICE


class Permute(nn.Module):
    """Base class for making permutations."""

    def __init__(self, *dims: Any) -> None:
        """Invoke parent's init method. Add attribute dimensions essential to further permutations."""

        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make next permutation of data.

        Args:
            x: tensor containing data to permutation.

        Returns:
            x: original tensor with its dimensions permuted.
        """

        return x.permute(self.dims)


class Transpose(nn.Module):
    """Base class for transposing data."""

    def __init__(self, *dims: Any, contiguous: bool = False) -> None:
        """Invoke parent's init method. Receive dimensions of data in tuple.
        Control, whether data is contiguous.
        """

        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method transposes given data. Checks, whether data is contiguous.
        The output data varies depending on this fact.

        Args:
            x: tensor containing data dimensions to transposing.

        Returns:
            x: original tensor with swapped data. Returned values are different for contiguous
            and not contiguous data.

        """

        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class TransformerModel(nn.Module):
    """Class sets up neural network which render context.
    This class is crucial for forecasting and working with data.
    """

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
            activation (Nonlin, optional):
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
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers_hidden, norm=encoder_norm)

        self.model = nn.Sequential(
            Permute(1, 0, 2),  # bs x seq_len x nvars -> seq_len x bs x nvars
            nn.Linear(n_units_in, n_units_hidden),  # seq_len x bs x nvars -> seq_len x bs x n_units_hidden
            nn.ReLU(),
            self.transformer_encoder,
            Transpose(1, 0),  # seq_len x bs x n_units_hidden -> bs x seq_len x n_units_hidden
            nn.ReLU(),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create new model. Make set of operations on given data.

        Args:
            x: tensor with raw data.

        Returns:
            model: original tensor which data have been packed in model layer and changed.
        """

        return self.model(x)  # pylint: disable=not-callable
