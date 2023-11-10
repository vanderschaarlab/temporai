"""Utilities for the ``models`` package directory."""

import os
import random
import warnings
from typing import Any, Dict, Type, Union

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data.sampler
import torch.version
from packaging.version import Version
from torch import nn

import tempor.core.utils

from .constants import DEVICE, Nonlin, Samp


def enable_reproducibility(
    random_seed: int = 0,
    torch_use_deterministic_algorithms: bool = False,
    torch_set_cudnn_deterministic: bool = False,
    torch_disable_cudnn_benchmark: bool = False,
    warn_cuda_env_vars: bool = True,
) -> None:
    """Attempt to enable reproducibility of results by removing sources of non-determinism (randomness) wherever
    possible. This function does not guarantee reproducible results, as there could be many other sources of
    randomness, e.g. data splitting, third party libraries etc.

    The implementation is based on the information in PyTorch documentation here:
    https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        random_seed (int, optional):
            The random seed to set. Defaults to 0.
        torch_use_deterministic_algorithms (bool, optional):
            Whether to set ``torch.use_deterministic_algorithms(True)``. Defaults to `False`.
        torch_set_cudnn_deterministic (bool, optional):
            Whether to set ``torch.backends.cudnn.deterministic = True``. Defaults to `False`.
        torch_disable_cudnn_benchmark (bool, optional):
            Whether to set ``torch.backends.cudnn.benchmark = False``. Defaults to `False`.
        warn_cuda_env_vars (bool, optional):
            Whether to raise a `RuntimeWarning` in case `torch` deterministic algorithms are enabled but the
            ``"CUDA_LAUNCH_BLOCKING"``/``"CUBLAS_WORKSPACE_CONFIG"`` environment variable has not been set.
            More details at https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM.
            Defaults to `True`.
    """
    # Built-in random module.
    random.seed(random_seed)

    # NumPy:
    np.random.seed(random_seed)

    # PyTorch:
    # Main seed:
    torch.manual_seed(random_seed)
    # Cuda seed, even if multiple GPUs:
    torch.cuda.manual_seed_all(random_seed)
    # If enabled, force deterministic algorithms:
    if torch_use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)  # type: ignore [no-untyped-call]
        if warn_cuda_env_vars and torch.version.cuda not in ("", None):
            cuda_version = Version(torch.version.cuda)
            if (cuda_version.major, cuda_version.minor) == (10, 1) and os.environ.get(
                "CUDA_LAUNCH_BLOCKING", None
            ) != "1":
                warnings.warn(
                    "When setting torch.use_deterministic_algorithms and using CUDA 10.1, the environment variable "
                    "CUDA_LAUNCH_BLOCKING must be set to 1, else RNN/LSTM algorithms will not be deterministic.",
                    RuntimeWarning,
                )
            if cuda_version >= Version("10.2") and (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG", None) not in (":4096:2", ":16:8")
            ):
                warnings.warn(
                    "When setting torch.use_deterministic_algorithms and using CUDA 10.2 or later, the environment "
                    "variable CUBLAS_WORKSPACE_CONFIG must be set to :4096:2 or :16:8, else RNN/LSTM algorithms will "
                    "not be deterministic.",
                    RuntimeWarning,
                )
    # If enabled, set the CuDNN deterministic option.
    if torch_set_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    # If enabled, disable CuDNN benchmarking process to avoid possible non-determinism:
    if torch_disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False


class GumbelSoftmax(nn.Module):
    def __init__(self, tau: float = 0.2, hard: bool = False, dim: int = -1) -> None:
        """Gumbel-Softmax activation function implementation."""
        super(GumbelSoftmax, self).__init__()

        self.tau = tau
        self.hard = hard
        self.dim = dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return nn.functional.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=self.dim)


NONLIN_MAP: Dict[str, nn.Module] = {
    "none": nn.Identity(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=-1),
    "gumbel_softmax": GumbelSoftmax(dim=-1),
}
"""A map of names (str, keys) to nonlinearity modules (`nn.Module`, values)."""
tempor.core.utils.ensure_literal_matches_dict_keys(Nonlin, NONLIN_MAP, "Nonlin", "NONLIN_MAP")


def get_nonlin(name: Nonlin) -> nn.Module:
    """Get a nonlinearity `nn.Module` (nonlinearity / activation function) by name.

    Args:
        name (Nonlin): Nonlinearity name.

    Raises:
        ValueError: If unknown nonlinearity name.

    Returns:
        nn.Module: Nonlinearity module.
    """
    try:
        return NONLIN_MAP[name]
    except KeyError as e:
        raise ValueError(f"Unknown nonlinearity {name}") from e


SAMPLER_MAP: Dict[str, Type[torch.utils.data.sampler.Sampler]] = {
    "BatchSampler": torch.utils.data.sampler.BatchSampler,
    "RandomSampler": torch.utils.data.sampler.RandomSampler,
    "Sampler": torch.utils.data.sampler.Sampler,
    "SequentialSampler": torch.utils.data.sampler.SequentialSampler,
    "SubsetRandomSampler": torch.utils.data.sampler.SubsetRandomSampler,
    "WeightedRandomSampler": torch.utils.data.sampler.WeightedRandomSampler,
}
"""A map of names (str, keys) to sampler classes (`Type[torch.utils.data.sampler.Sampler]`, values)."""
tempor.core.utils.ensure_literal_matches_dict_keys(Samp, SAMPLER_MAP, "Samp", "SAMPLER_MAP")


def get_sampler(name: Union[Samp, None], **kwargs: Any) -> Union[torch.utils.data.sampler.Sampler, None]:
    """Get a sampler by name.

    Args:
        name (Union[Samp, None]): Sampler name.
        **kwargs (Any): Sampler initializer kwargs.

    Raises:
        ValueError: If unknown sampler name.

    Returns:
        Union[torch.utils.data.sampler.Sampler, None]: Sampler instance.
    """
    try:
        return SAMPLER_MAP[name](**kwargs) if name is not None else None
    except KeyError as e:
        raise ValueError(f"Unknown sampler {name}") from e


def get_device(device: Union[None, int, str, torch.device]) -> torch.device:
    """Get a `torch` device by name.

    Args:
        device (Union[None, int, str, torch.device]):
            Arguments to pass to `torch.device`, or `None` to use the default device.

    Returns:
        torch.device: Device instance.
    """
    if device is None:
        return DEVICE
    else:
        return torch.device(device)
