from typing import NoReturn, Tuple

from ..data import DEFAULT_PADDING_INDICATOR
from . import tensor_like as tl
from .tensor_like import TTensorLike


def _raise_wrong_dim(name, ndim) -> NoReturn:
    raise ValueError(f"`{name}` must have 3 dimensions but {ndim} found")


def validate_not_all_padding(tensor: TTensorLike, padding_indicator: float) -> None:
    if tensor.ndim != 3:
        _raise_wrong_dim("tensor", tensor.ndim)
    bools_array = tl.eq_indicator(tensor, padding_indicator)
    outcome = tl.any(tl.all(tl.any(bools_array, axis=-1), axis=-1))
    if outcome:
        raise ValueError(
            "Encountered an array/tensor with all padding values along time dimension (dimension 1), "
            "for at least one sample (dimension 0)."
        )


def n_step_shift_back(
    shift_back: TTensorLike,
    n_step: int,
    padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    validate_not_all_padding_: bool = True,
):
    shift_back_str = "shift_back"
    if shift_back.ndim != 3:
        _raise_wrong_dim(shift_back_str, shift_back.ndim)
    if n_step >= shift_back.shape[1]:
        raise ValueError(
            f"Size of dimension 1 (time dimension) of `{shift_back_str}` ({shift_back.shape[1]}) "
            f"is too short to shift by {n_step} step(s)"
        )

    # Shift:
    shift_back_shifted: TTensorLike = shift_back[:, n_step:, :]

    # Check all-padding case (none of the samples must be all padding):
    if validate_not_all_padding_:
        validate_not_all_padding(shift_back_shifted, padding_indicator)

    return shift_back_shifted


def n_step_shift_forward(
    shift_forward: TTensorLike,
    n_step: int,
    padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    validate_not_all_padding_: bool = True,
):
    shift_forward_str = "shift_forward"
    if shift_forward.ndim != 3:
        _raise_wrong_dim(shift_forward_str, shift_forward.ndim)
    if n_step >= shift_forward.shape[1]:
        raise ValueError(
            f"Size of dimension 1 (time dimension) of `{shift_forward_str}` ({shift_forward.shape[1]}) "
            f"is too short to shift by {n_step} step(s)"
        )

    # Shift:
    shift_forward_shifted: TTensorLike = shift_forward[:, :-n_step, :]

    # Check all-padding case (none of the samples must be all padding):
    if validate_not_all_padding_:
        validate_not_all_padding(shift_forward_shifted, padding_indicator)

    return shift_forward_shifted


def n_step_shifted(
    shift_back: TTensorLike,
    shift_forward: TTensorLike,
    n_step: int,
    padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    validate_not_all_padding_: bool = True,
) -> Tuple[TTensorLike, TTensorLike]:
    # Validate:
    shift_back_str = "shift_back"
    shift_forward_str = "shift_forward"

    # Shift:
    shift_forward_shifted: TTensorLike = n_step_shift_forward(
        shift_forward, n_step, padding_indicator, validate_not_all_padding_
    )
    shift_back_shifted: TTensorLike = n_step_shift_back(
        shift_back, n_step, padding_indicator, validate_not_all_padding_
    )

    # Validation delayed, just to make sure 3 dims first.
    if shift_forward.shape[1] != shift_back.shape[1]:
        raise ValueError(
            f"`{shift_back_str}` and each `{shift_forward_str}` must have equal size of dimension 1 (time dimension). "
            f"But was: {shift_back.shape[1]} for `{shift_back_str}` and {shift_forward.shape[1]} "
            f"in `{shift_forward_str}`"
        )

    return shift_back_shifted, shift_forward_shifted


def compute_deltas(tensor: TTensorLike, padding_indicator: float = DEFAULT_PADDING_INDICATOR) -> TTensorLike:
    if tensor.ndim != 3:
        _raise_wrong_dim("tensor", tensor.ndim)
    out = tl.zeros_like(tensor)
    out[:, 1:, :] = tl.diff(tensor, axis=1)
    out[tl.eq_indicator(tensor, padding_indicator)] = padding_indicator
    return out
