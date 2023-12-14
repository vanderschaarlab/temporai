from collections import namedtuple
from typing import Dict, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..data import Dataset, StaticSamples, TimeSeriesSamples
from ..data.constants import DEFAULT_PADDING_INDICATOR
from ..utils.dev import raise_not_implemented


def _to_tensor_handle_none(
    array: Optional[np.ndarray], nans_shape: Optional[Tuple[int, ...]] = None, dtype=torch.float
) -> torch.Tensor:
    if array is not None:
        return torch.tensor(array, dtype=dtype)
    else:
        if nans_shape is None:
            raise ValueError("Must provide `nans_shape` if `array` is None")
        else:
            return torch.full(nans_shape, fill_value=torch.nan, dtype=dtype)


# TODO: Remove this.
class ClairvoyanceTorchDataset(TorchDataset):
    container_defs = {
        "temporal_covariates": TimeSeriesSamples,
        "static_covariates": StaticSamples,
        "temporal_targets": TimeSeriesSamples,
        "temporal_treatments": TimeSeriesSamples,
    }

    def __init__(
        self,
        data: Dataset,
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
        max_len: Optional[int] = None,
        torch_dtype: torch.dtype = torch.float,
    ) -> None:
        assert all(x in dir(data) for x in self.container_defs.keys()), "Not all container names found in data"
        self.padding_indicator = padding_indicator
        self.max_len = max_len
        self.torch_dtype = torch_dtype
        self._initialize_torch_tensors(data)
        super().__init__()

    def _initialize_torch_tensors(self, data: Dataset) -> None:
        self.torch_tensors: Dict[str, torch.Tensor] = dict()
        for container_name in self.container_defs.keys():
            array: Optional[np.ndarray] = None
            nans_shape: Optional[Tuple] = None
            if container_name in data.all_data_containers:
                container = data.all_data_containers[container_name]
                if isinstance(container, StaticSamples):
                    array = container.to_numpy()
                elif isinstance(container, TimeSeriesSamples):
                    array = container.to_numpy(padding_indicator=self.padding_indicator, max_len=self.max_len)
                else:
                    raise_not_implemented(f"ClairvoyanceTorchDataset with {container.__class__.__name__} containers")
            else:
                if self.container_defs[container_name] == StaticSamples:
                    nans_shape = (data.n_samples, 1)
                else:
                    nans_shape = (data.n_samples, 1, 1)
            self.torch_tensors[container_name] = _to_tensor_handle_none(
                array, nans_shape=nans_shape, dtype=self.torch_dtype
            )
        assert list(self.container_defs.keys()) == list(self.torch_tensors.keys())

    def __len__(self) -> int:
        return list(self.torch_tensors.values())[0].shape[0]

    @staticmethod
    def _dispatch_getitem(tensor: torch.Tensor, idx: int) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor[idx, :, :]
        elif tensor.ndim == 2:
            return tensor[idx, :]
        else:  # pragma: no cover
            raise ValueError(f"Tensor had an unexpected number of dimensions {tensor.ndim}, expected 2 or 3")

    def __getitem__(self, idx: int):
        DatasetItem = namedtuple("DatasetItem", list(self.container_defs.keys()))  # type: ignore
        return DatasetItem(**{k: self._dispatch_getitem(t, idx) for k, t in self.torch_tensors.items()})


TContainer = Union[torch.Tensor, np.ndarray, TimeSeriesSamples, StaticSamples]


# NOTE: Currently not used.
class CustomTorchDataset(TorchDataset):
    def __init__(
        self,
        containers: Sequence[Optional[TContainer]],
        index_dimensions: Optional[Sequence[int]] = None,
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
        max_len: Optional[int] = None,
        torch_dtype: torch.dtype = torch.float,
    ) -> None:
        if len(containers) == 0:
            raise ValueError("Must provide at least one container")
        if all([c is None for c in containers]):
            self._raise_must_not_be_all_none()

        self.padding_indicator = padding_indicator
        self.max_len = max_len
        self.torch_dtype = torch_dtype

        if index_dimensions is None:
            index_dimensions = [0] * len(containers)
        self._validate(containers, index_dimensions)
        self.index_dimensions = index_dimensions

        self.n_samples = self._init_n_samples(containers)

        self._initialize_torch_tensors(containers)
        self._validate_len_index_dimensions()

        super().__init__()

    @staticmethod
    def _raise_must_not_be_all_none() -> NoReturn:
        raise ValueError("At least one container provided must not be None")

    @staticmethod
    def _raise_wrong_container_type() -> NoReturn:
        raise ValueError(f"`container` was not one of types: {TContainer}")

    @staticmethod
    def _get_ndim(container: Optional[TContainer]) -> int:
        if isinstance(container, (np.ndarray, torch.Tensor)):
            return container.ndim
        elif isinstance(container, TimeSeriesSamples):
            return 3
        elif isinstance(container, StaticSamples):
            return 2
        elif container is None:
            return 1
        else:
            CustomTorchDataset._raise_wrong_container_type()

    def _init_n_samples(self, containers: Sequence[Optional[TContainer]]) -> int:
        for idx, c in enumerate(containers):
            if c is not None:
                if isinstance(c, (StaticSamples, TimeSeriesSamples)):
                    return c.n_samples
                else:
                    idx_dim = self.index_dimensions[idx]
                    return c.shape[idx_dim]
            else:
                # To avoid confusing test coverage.
                pass  # pragma: no cover
        self._raise_must_not_be_all_none()  # pragma: no cover

    def _validate(self, containers: Sequence[Optional[TContainer]], index_dimensions: Sequence[int]) -> None:
        if len(index_dimensions) != len(containers):
            raise ValueError(
                "Length of `containers` sequence must be the same as the length of "
                "the `index_dimensions` sequence provided"
            )
        if not all([i_dim < self._get_ndim(c) for i_dim, c in zip(index_dimensions, containers)]):
            raise ValueError(
                "One of the index dimensions provided exceeded the number of dimensions of a container, "
                "check `index_dimensions`"
            )
        for i_dim, c in zip(index_dimensions, containers):
            if isinstance(c, (StaticSamples, TimeSeriesSamples)) and i_dim != 0:
                raise ValueError(
                    f"Can only index {StaticSamples.__name__} and "
                    f"{TimeSeriesSamples.__name__} by 0th dimension, check `index_dimensions`"
                )

    def _initialize_torch_tensors(self, containers: Sequence[Optional[TContainer]]) -> None:
        self.tensors: List[torch.Tensor] = []
        for c in containers:
            if isinstance(c, torch.Tensor):
                self.tensors.append(c.to(dtype=self.torch_dtype))
            elif isinstance(c, np.ndarray):
                self.tensors.append(torch.tensor(c).to(dtype=self.torch_dtype))
            elif isinstance(c, StaticSamples):
                self.tensors.append(torch.tensor(c.to_numpy()).to(dtype=self.torch_dtype))
            elif isinstance(c, TimeSeriesSamples):
                self.tensors.append(
                    torch.tensor(c.to_numpy(padding_indicator=self.padding_indicator, max_len=self.max_len)).to(
                        dtype=self.torch_dtype
                    )
                )
            elif c is None:
                self.tensors.append(torch.full((self.n_samples,), fill_value=torch.nan, dtype=self.torch_dtype))
            else:  # pragma: no cover
                self._raise_wrong_container_type()

    def _get_len(self) -> int:
        return self.tensors[0].shape[self.index_dimensions[0]]

    def _validate_len_index_dimensions(self):
        expected_len = self._get_len()
        if not all(t.shape[i_dim] == expected_len for i_dim, t in zip(self.index_dimensions, self.tensors)):
            raise ValueError("Not all containers were the same length along the index dimension")

    def __len__(self) -> int:
        return self._get_len()

    def _get_slices(self, tensor_index: int, getitem_index: int) -> List[Union[slice, int]]:
        slices: List[Union[slice, int]] = [slice(None)] * self.tensors[tensor_index].ndim
        slices[self.index_dimensions[tensor_index]] = getitem_index
        return slices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return tuple(t[self._get_slices(tensor_idx, idx)] for tensor_idx, t in enumerate(self.tensors))


def to_custom_torch_dataset(
    containers: Sequence[Optional[TContainer]],
    index_dimensions: Optional[Sequence[int]] = None,
    padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    max_len: Optional[int] = None,
    torch_dtype: torch.dtype = torch.float,
) -> CustomTorchDataset:
    return CustomTorchDataset(
        containers=containers,
        index_dimensions=index_dimensions,
        padding_indicator=padding_indicator,
        max_len=max_len,
        torch_dtype=torch_dtype,
    )
