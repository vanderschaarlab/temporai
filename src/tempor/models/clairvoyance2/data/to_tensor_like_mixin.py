from typing import Optional, overload

import numpy as np
import torch

from .constants import DEFAULT_PADDING_INDICATOR


class ToTensorLikeMixin:
    @overload
    def to_numpy(self) -> np.ndarray:
        ...

    @overload
    def to_numpy(
        self, *, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        ...

    def to_numpy(self, *, padding_indicator: Optional[float] = None, max_len: Optional[int] = None) -> np.ndarray:
        if padding_indicator is None and max_len is None:
            try:
                return self._to_numpy_static()
            except NotImplementedError:
                return self._to_numpy_time_series(padding_indicator=DEFAULT_PADDING_INDICATOR, max_len=None)
                # ^ Call with default arguments.
        else:
            if padding_indicator is None:
                padding_indicator = DEFAULT_PADDING_INDICATOR
            if not isinstance(padding_indicator, float):
                raise TypeError("`padding_indicator` must be a float")
            if not (isinstance(max_len, int) or max_len is None):
                raise TypeError("`max_len` must be an int or None")
            return self._to_numpy_time_series(padding_indicator=padding_indicator, max_len=max_len)

    def to_numpy_time_index(
        self, *, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        return self._to_numpy_time_index(padding_indicator=padding_indicator, max_len=max_len)

    def _to_numpy_time_series(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        raise NotImplementedError("`_to_numpy_time_series` method not implemented")

    def _to_numpy_static(self) -> np.ndarray:
        raise NotImplementedError("`_to_numpy_static` method not implemented")

    def _to_numpy_time_index(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        raise NotImplementedError("`_to_numpy_time_index` method not implemented")

    @overload
    def to_torch_tensor(self, **torch_tensor_kwargs) -> torch.Tensor:
        ...

    @overload
    def to_torch_tensor(  # type: ignore  # For clarity only.
        self,
        *,
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
        max_len: Optional[int] = None,
        **torch_tensor_kwargs,
    ) -> torch.Tensor:
        ...

    def to_torch_tensor(
        self, *, padding_indicator: Optional[float] = None, max_len: Optional[int] = None, **torch_tensor_kwargs
    ) -> torch.Tensor:
        return torch.tensor(
            self.to_numpy(padding_indicator=padding_indicator, max_len=max_len), **torch_tensor_kwargs  # type: ignore
        )

    def to_torch_tensor_time_index(
        self,
        *,
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
        max_len: Optional[int] = None,
        **torch_tensor_kwargs,
    ) -> torch.Tensor:
        return torch.tensor(
            self.to_numpy_time_index(padding_indicator=padding_indicator, max_len=max_len), **torch_tensor_kwargs
        )
