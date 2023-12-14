from typing import Callable, NamedTuple, Sequence, TypeVar, Union

import numpy as np
import torch

TTensorLike = TypeVar("TTensorLike", np.ndarray, torch.Tensor)
TTensorLike_AsTuple = tuple([np.ndarray, torch.Tensor])

TTensorLikeOrBool = Union[TTensorLike, bool, np.bool_]
TTensorLikeOrBool_AsTuple = tuple([*TTensorLike_AsTuple, bool, np.bool_])

# TODO: Tidy this module.


class FunctionMap(NamedTuple):
    np_func: Callable
    torch_func: Callable


def _kwargs_process(kwargs):
    if "array" in kwargs:
        kwargs["dim"] = kwargs["array"]
        del kwargs["array"]
    return kwargs


def _typical_function_call(function_map: FunctionMap, a: TTensorLike, **kwargs) -> TTensorLike:
    if isinstance(a, torch.Tensor):
        kwargs = _kwargs_process(kwargs)
        return function_map.torch_func(a, **kwargs)
    else:
        return function_map.np_func(a, **kwargs)


def _typical_function_call_sequence(function_map: FunctionMap, a_seq: Sequence[TTensorLike], **kwargs) -> TTensorLike:
    if isinstance(a_seq[0], torch.Tensor):
        kwargs = _kwargs_process(kwargs)
        return function_map.torch_func(a_seq, **kwargs)
    else:
        return function_map.np_func(a_seq, **kwargs)


def _all_or_any(all_or_any: str, a: TTensorLikeOrBool, **kwargs):
    assert isinstance(a, TTensorLikeOrBool_AsTuple)
    if isinstance(a, bool):
        return a
    if all_or_any == "all":
        fm = FunctionMap(np_func=np.all, torch_func=torch.all)
    else:
        fm = FunctionMap(np_func=np.any, torch_func=torch.any)
    if isinstance(a, torch.Tensor):
        kwargs = _kwargs_process(kwargs)
        return fm.torch_func(a, **kwargs)
    else:
        return fm.np_func(a, **kwargs)


def all(a: TTensorLikeOrBool, **kwargs) -> Union[TTensorLike, bool]:  # pylint: disable=redefined-builtin
    return _all_or_any("all", a, **kwargs)


def any(a: TTensorLikeOrBool, **kwargs) -> Union[TTensorLike, bool]:  # pylint: disable=redefined-builtin
    return _all_or_any("any", a, **kwargs)


def isnan(a: TTensorLike) -> TTensorLike:
    assert isinstance(a, TTensorLike_AsTuple)
    if isinstance(a, torch.Tensor):
        return torch.isnan(a)
    else:
        return np.isnan(a)


def eq_indicator(a: TTensorLike, indicator: float) -> TTensorLike:
    # Indicator may be nan.
    assert isinstance(a, TTensorLike_AsTuple)
    if np.isnan(indicator):
        return isnan(a)
    else:
        return a == indicator


def concatenate(tensors: Sequence[TTensorLike], **kwargs) -> TTensorLike:
    assert len(tensors) > 0
    return _typical_function_call_sequence(FunctionMap(np_func=np.concatenate, torch_func=torch.cat), tensors, **kwargs)


def zeros_like(a: TTensorLike, **kwargs) -> TTensorLike:
    return _typical_function_call(FunctionMap(np_func=np.zeros_like, torch_func=torch.zeros_like), a, **kwargs)


def diff(a: TTensorLike, **kwargs) -> TTensorLike:
    return _typical_function_call(FunctionMap(np_func=np.diff, torch_func=torch.diff), a, **kwargs)
