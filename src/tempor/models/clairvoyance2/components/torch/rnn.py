# mypy: ignore-errors

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...utils import tensor_like as tl
from ...utils.common import safe_init_dotmap
from ...utils.dev import raise_not_implemented
from .ffnn import FeedForwardNet

RNNClass = Type[nn.RNNBase]
RNNHidden = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

RNN_CLASS_MAP: Mapping[str, Type[nn.RNNBase]] = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}

_DEBUG = False


class RecurrentNet(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        nonlinearity: Optional[str],
        num_layers: int,
        bias: bool,
        dropout: float,
        bidirectional: bool,
        proj_size: Optional[int],
    ) -> None:
        super().__init__()

        kwargs: Dict[str, Any] = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,  # NOTE: We adopt batch first convention.
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.rnn_type = rnn_type
        rnn_class = RNN_CLASS_MAP[rnn_type]
        if rnn_class == nn.RNN:
            if nonlinearity is None:
                nonlinearity = "tanh"
            kwargs["nonlinearity"] = nonlinearity
        if rnn_class == nn.LSTM:
            if proj_size is None:
                proj_size = 0
            kwargs["proj_size"] = proj_size
        self.params = safe_init_dotmap(kwargs)

        self.rnn = rnn_class(**kwargs)

    def forward(self, x: torch.Tensor, h: Optional[RNNHidden]) -> Tuple[torch.Tensor, RNNHidden]:
        if h is not None:
            rnn_out, h_out = self.rnn(x, h)
        else:
            rnn_out, h_out = self.rnn(x)
        return rnn_out, h_out

    def get_output_and_h_dim(self) -> Tuple[int, Tuple[int, int, int]]:
        """A convenience method that computes the size of the output of `forward()` for each time-step.
        Useful for defining the input size of a downstream module to be applied at each timestep.

        For logic behind this calculation see:
        * https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        * https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        * https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

        Note:
            This class always has `batch_first=True`, so timesteps are in the second dimension of the output.

        Returns:
            Tuple[int, Tuple[int, int, int]]: (`output` feature dimension, (`D * num_layers`, `H_out`, `H_cell`)
        """
        if "proj_size" in self.params and self.params.proj_size > 0:
            h_out = self.params.proj_size
        else:
            h_out = self.params.hidden_size
        d = int(self.params.bidirectional) + 1
        out_dim = d * h_out
        d_num_layers = d * self.params.num_layers
        h_cell = 0
        if isinstance(self.rnn, nn.LSTM):
            h_cell = self.params.hidden_size
        if _DEBUG is True:  # pragma: no cover
            print("------ compute_rnn_output_dim_per_timestep() ------")
            print("h_out", h_out)
            print("d", d)
            print("out_dim", out_dim)
            print("d_num_layers", d_num_layers)
            print("h_cell", h_cell)
            print("--- compute_rnn_output_dim_per_timestep() [END] ---")
        return out_dim, (d_num_layers, h_out, h_cell)


class AutoregressiveMixin(ABC):
    def __init__(self, feed_first_n: Optional[int] = None) -> None:
        self.feed_first_n = feed_first_n
        self.x_used_in_autoregress: Any = None

    @abstractmethod
    def _forward_for_autoregress(
        self, x: torch.Tensor, timestep_idx: int, **kwargs
    ) -> torch.Tensor:  # pragma: no cover
        ...

    @staticmethod
    def _validate_shape(t: torch.Tensor, t_name: str, feed_first_n: Optional[int]) -> None:
        if t.ndim != 3:
            raise RuntimeError(f"{t_name} expected to have 3 dimensions but {t.ndim} found")
        if feed_first_n is not None:
            if feed_first_n > t.shape[-1]:
                raise RuntimeError(
                    f"`feed_first_n` ({feed_first_n}) must be < or = the size "
                    f"of the last dimension of {t_name} ({t.shape[-1]})"
                )

    def autoregress(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self._validate_shape(x, "`x`", self.feed_first_n)
        self.x_used_in_autoregress = x.clone()
        out_list: List[torch.Tensor] = []
        n_timesteps = x.shape[1]
        for time_idx in range(n_timesteps):
            out = self._forward_for_autoregress(self.x_used_in_autoregress[:, [time_idx], :], time_idx, **kwargs)
            self._validate_shape(out, "`forward_for_autoregress()` output", self.feed_first_n)
            assert out.shape[1] == 1
            if self.feed_first_n is None:
                if out.shape[-1] != x.shape[-1]:
                    raise RuntimeError(
                        "`forward_for_autoregress()` output and `x` last dimension must be the same size "
                        f"but were {out.shape[-1]} and {x.shape[-1]} respectively"
                    )
            out_list.append(out)
            if time_idx < n_timesteps - 1:
                self.x_used_in_autoregress[:, [time_idx + 1], : self.feed_first_n] = out
        out = torch.cat(out_list, dim=1)
        return out


class RecurrentFFNet(AutoregressiveMixin, nn.Module):
    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        nonlinearity: Optional[str],
        num_layers: int,
        bias: bool,
        dropout: float,
        bidirectional: bool,
        proj_size: Optional[int],
        # ---
        ff_out_size: int,
        ff_in_size_adjust: int = 0,
        ff_hidden_dims: Sequence[int] = tuple(),
        ff_out_activation: Optional[str] = "ReLU",
        ff_hidden_activations: Optional[str] = "ReLU",
    ) -> None:
        nn.Module.__init__(self)
        AutoregressiveMixin.__init__(self, feed_first_n=ff_out_size)
        self.rnn_type = rnn_type
        self.rnn = RecurrentNet(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )
        self.ff_in_size, *_ = self.rnn.get_output_and_h_dim()
        self.ff_in_size += ff_in_size_adjust
        self.ff_out_size = ff_out_size
        self.ffnn = FeedForwardNet(
            in_dim=self.ff_in_size,
            out_dim=self.ff_out_size,
            hidden_dims=ff_hidden_dims,
            out_activation=ff_out_activation,
            hidden_activations=ff_hidden_activations,
        )

    def rnn_out_postprocess(self, rnn_out: torch.Tensor, **kwargs) -> torch.Tensor:  # pylint: disable=unused-argument
        return rnn_out

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[RNNHidden],
        padding_indicator: Optional[float] = None,
        **kwargs_rnn_out_callback,
    ) -> Tuple[torch.Tensor, torch.Tensor, RNNHidden]:
        if padding_indicator is not None:
            with packed(x, padding_indicator) as p:
                p.packed, h = self.rnn(p.packed, h=h)
            rnn_out = p.unpacked
        else:
            rnn_out, h = self.rnn(x, h=h)

        if TYPE_CHECKING:
            assert h is not None

        if _DEBUG is True:  # pragma: no cover
            print("rnn_out.shape", rnn_out.shape)  # type: ignore
            print("h (or h concat c) shape", rnn_out.shape)  # type: ignore

        # TODO: Possibly an option to concatenate *non* last layer's h_n[, c_n], but may be needlessly complex.
        # h_flattened = h.reshape(shape=[current_batch_size, -1])
        # print("h_flattened.shape", h_flattened.shape)

        rnn_out_postprocessed = self.rnn_out_postprocess(rnn_out, **kwargs_rnn_out_callback)  # type: ignore

        out = apply_to_each_timestep(
            self.ffnn,
            input_tensor=rnn_out_postprocessed,
            output_size=self.ff_out_size,
            concat_tensors=[],
            padding_indicator=padding_indicator,
            expected_module_input_size=self.ff_in_size,
        )
        return out, rnn_out, h  # type: ignore

    def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
        out, *_ = self.forward(x, **kwargs)
        return out


@dataclass
class PackedContainer:
    packed: torch.nn.utils.rnn.PackedSequence
    unpacked: Optional[torch.Tensor] = None
    unpacked_lens: Optional[torch.Tensor] = None


@contextlib.contextmanager
def packed(x: torch.Tensor, padding_indicator: float, batch_first: bool = True, enforce_sorted: bool = False):
    if x.ndim != 3:
        raise RuntimeError(f"Input to `packed` must be a 3 dimensional tensor but {x.ndim} dimensions found")
    if batch_first is False:
        raise_not_implemented("`packed()` with batch_first = False")

    max_len = x.shape[1]
    # Treat as padding if *any* of the features has a padding value:
    padding_bools = tl.eq_indicator(x.detach(), padding_indicator).any(dim=-1)
    # Assert that all padding was at the end:
    expect_padding_true = padding_bools.sum(dim=1)
    for idx, len_ in enumerate(expect_padding_true):
        if len_ > 0:
            if (padding_bools[idx, -len_:] == False).any():  # noqa: E712
                raise RuntimeError("Found padding values not at the end of sequences")

    where_all_padding = padding_bools.all(dim=1)

    lengths_type = torch.int64
    device = where_all_padding.device

    out_lens_template = torch.zeros(size=where_all_padding.shape, dtype=lengths_type, device=device)
    x_exclude_all_padding_samples = x[~where_all_padding, :, :]

    x_seq_lens = (~padding_bools).sum(dim=1)[~where_all_padding]
    x_seq_lens = x_seq_lens.to(device="cpu", dtype=lengths_type)
    x_packed = pack_padded_sequence(
        x_exclude_all_padding_samples, x_seq_lens, batch_first=batch_first, enforce_sorted=enforce_sorted
    )
    packed_container = PackedContainer(x_packed)

    try:
        yield packed_container

    finally:
        x_unpacked, x_unpacked_lens = pad_packed_sequence(
            packed_container.packed, batch_first=batch_first, padding_value=padding_indicator, total_length=max_len
        )
        out_template = torch.full(
            size=(x.shape[0], x.shape[1], x_unpacked.shape[2]),
            fill_value=padding_indicator,
            dtype=x_unpacked.dtype,
            device=x_unpacked.device,
        )
        out_template[~where_all_padding, :, :] = x_unpacked
        out_lens_template[~where_all_padding] = x_unpacked_lens.to(device=device)
        packed_container.unpacked = out_template
        packed_container.unpacked_lens = out_lens_template


def apply_to_each_timestep(
    module: nn.Module,
    input_tensor: torch.Tensor,
    output_size: int,
    expected_module_input_size: int,
    padding_indicator: Optional[float],
    concat_tensors: Iterable[torch.Tensor] = tuple(),
) -> torch.Tensor:
    """Applies `module` forward to each timestep of `input_tensor`. Timestep dimension is expected to be dimension 1.

    Args:
        module (`nn.Module`): Module to apply at each timestep.
        input_tensor (`torch.Tensor`): Tensor to apply module to. Shape: `[n_samples, n_timesteps, n_features]`.
        output_size (`int`): The size of the feature (last) dimension of the `module` output.
        expected_module_input_size (`int`): Will check that module input dimension is this value.
        padding_indicator (`Optional[float]`): If `None`, assume no padding in `input_tensor` timestep dimension. If a
            float value (or `nan`), those tensor elements are treated as padding and not passed through `module`.
        concat_tensors (`Iterable[torch.Tensor]`): Optionally provide a sequence of tensors to concatenate to input at
            each timestep, before passing to `module`. Defaults to `tuple()`.

    Raises:
        `RuntimeError`: If `expected_module_input_size` input size check fails.

    Returns:
        `torch.Tensor`: Output tensor.
    """
    assert module is not None

    module_out_list = []

    for timestep_idx in range(input_tensor.shape[1]):
        input_timestep = input_tensor[:, timestep_idx, :]
        module_in = torch.cat([input_timestep, *concat_tensors], dim=-1)
        if _DEBUG is True:  # pragma: no cover
            print("input_timestep.shape", input_timestep.shape)
            print("module_in.shape", module_in.shape)

        fill_val = 0.0
        if padding_indicator is not None:
            is_padding_selector = tl.eq_indicator(input_timestep[:, -1], padding_indicator)
            assert isinstance(is_padding_selector, torch.Tensor)
            module_in = module_in[~is_padding_selector]
            fill_val = padding_indicator
        module_out_template = torch.full(
            size=(input_timestep.shape[0], output_size), fill_value=fill_val, device=input_tensor.device
        )
        if _DEBUG is True:  # pragma: no cover
            print("module_out_template.shape", module_out_template.shape)

        if module_in.shape[-1] != expected_module_input_size:
            raise RuntimeError(
                f"Module input wasn't of expected size, expected {expected_module_input_size}, "
                f"was {module_in.shape[-1]}"
            )

        module_out_timestep = module(module_in)
        if _DEBUG is True:  # pragma: no cover
            print("module_out_timestep.shape", module_out_timestep.shape)

        # Overwrite with padding value:
        # TODO: better way of masking?
        if padding_indicator is not None:
            assert isinstance(is_padding_selector, torch.Tensor)  # type: ignore
            module_out_template[~is_padding_selector] = module_out_timestep
        else:
            module_out_template[:] = module_out_timestep

        module_out_list.append(module_out_template)

    # Concatenate along the time dimension, to get shape (n_samples, n_timesteps, output_size)
    final_output = torch.stack(module_out_list, dim=1)
    if _DEBUG is True:  # pragma: no cover
        print("final_output.shape", final_output.shape)

    return final_output


def mask_and_reshape(mask_selector: torch.BoolTensor, tensor: torch.Tensor) -> torch.Tensor:
    # First applies `mask_selector` selector to `tensor` to take only values where `mask_selector` is True, this makes
    # a 1D tensor. Then reshapes this resultant tensor to have the same size on the last dimension as the original
    # tensor, as in: tensor_masked.reshape(-1, tensor.shape[-1])
    tensor_masked = torch.masked_select(tensor, mask=mask_selector)
    tensor_final = tensor_masked.reshape(-1, tensor.shape[-1])
    return tensor_final
