import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn
import torch.version


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
            Whether to raise a `UserWarning` in case `torch` deterministic algorithms are enabled but the
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
        torch.use_deterministic_algorithms(True)
        if warn_cuda_env_vars and torch.version.cuda not in ("", None):
            major, minor = [int(x) for x in torch.version.cuda.split(".")]
            if (major, minor) == (10, 1) and os.environ.get("CUDA_LAUNCH_BLOCKING", None) != "1":
                warnings.warn(
                    "When setting torch.use_deterministic_algorithms and using CUDA 10.1, the environment variable "
                    "CUDA_LAUNCH_BLOCKING must be set to 1, else RNN/LSTM algorithms will not be deterministic.",
                    UserWarning,
                )
            if (
                major > 10
                or (major == 10 and minor >= 2)
                and os.environ.get("CUBLAS_WORKSPACE_CONFIG", None) not in (":4096:2", ":16:8")
            ):
                warnings.warn(
                    "When setting torch.use_deterministic_algorithms and using CUDA 10.2 or later, the environment "
                    "variable CUBLAS_WORKSPACE_CONFIG must be set to :4096:2 or :16:8, else RNN/LSTM algorithms will "
                    "not be deterministic.",
                    UserWarning,
                )
    # If enabled, set the CuDNN deterministic option.
    if torch_set_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    # If enabled, disable CuDNN benchmarking process to avoid possible non-determinism:
    if torch_disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
